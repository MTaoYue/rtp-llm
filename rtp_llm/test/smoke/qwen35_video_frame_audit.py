"""Manual decoded-frame integrity diagnostic; separate from performance measurement."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace


def frame_hashes(video):
    return [hashlib.sha256(frame.contiguous().numpy().tobytes()).hexdigest() for frame in video]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--video', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--iterations', type=int, default=256)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    if a.iterations < 1 or a.workers < 1:
        raise ValueError('positive diagnostic sizes required')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    import torch
    from transformers import AutoProcessor
    from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe import gpu_video
    from rtp_llm.multimodal.qwen3_vl_video import decode_video, resolve_video_size
    torch.set_num_threads(1)
    encoded = Path(a.video).read_bytes()
    processor = AutoProcessor.from_pretrained(a.model_dir, local_files_only=True).video_processor
    config = SimpleNamespace(fps=6, min_frames=4, max_frames=180)
    data, grid = gpu_video.prepare_gpu_video(encoded, config, processor, 32,
                                             resolve_video_size(processor, 2500000, 73728000))
    assert len(data.frame_indices) == 46 and grid.tolist() == [[23, 44, 80]], grid
    device = torch.device('cuda', 0)
    reference = gpu_video._decode_video_ready(data, device).cpu()
    reference_hashes = frame_hashes(reference)
    cpu, metadata = decode_video(io.BytesIO(encoded), config)
    assert tuple(cpu.shape) == tuple(reference.shape)
    assert tuple(metadata.frames_indices) == data.frame_indices
    delta = (reference.to(torch.int16)-cpu.to(torch.int16)).abs()
    result = dict(status='RUNNING', video_sha256=hashlib.sha256(encoded).hexdigest(),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  shape=list(reference.shape), grid=grid.tolist(), frame_indices=list(data.frame_indices),
                  source_fps=data.source_fps, reference_frame_sha256=reference_hashes,
                  cpu_frame_sha256=frame_hashes(cpu),
                  cpu_comparison=dict(exact_match=bool(torch.equal(reference,cpu)), max_abs=int(delta.max()), mean_abs=float(delta.float().mean()),
                                      differing_values=int((delta != 0).sum())),
                  iterations=a.iterations, workers=a.workers, mismatches=[],
                  gpu_video_source_sha256=hashlib.sha256(Path(gpu_video.__file__).read_bytes()).hexdigest(),
                  note='PASSED means repeated NVDEC RGB frames match the first NVDEC sample. CPU equality is reported separately. Hash copies synchronize and this diagnostic is not a throughput benchmark.')
    (out/'result.json').write_text(json.dumps(result,indent=2))
    pool = gpu_video.VideoDecodePool(a.workers)
    try:
        with (out/'frame-hashes.jsonl').open('x') as journal:
            with pool.decode((data for _ in range(a.iterations)), device) as videos:
                for index, video in enumerate(videos):
                    host = video.cpu()
                    hashes = frame_hashes(host)
                    mismatch = [n for n,(x,y) in enumerate(zip(hashes,reference_hashes)) if x != y]
                    record = dict(index=index, frame_sha256=hashes, mismatch_frames=mismatch)
                    if mismatch:
                        diff = (host.to(torch.int16)-reference.to(torch.int16)).abs()
                        record.update(max_abs=int(diff.max()), differing_values=int((diff != 0).sum()))
                        result['mismatches'].append({k:v for k,v in record.items() if k!='frame_sha256'})
                        if len(result['mismatches']) == 1:
                            # Preserve the first differing decoded sample for diagnosis.
                            torch.save(dict(reference=reference, actual=host),out/'first-frame-mismatch.pt')
                    journal.write(json.dumps(record)+'\n');journal.flush()
                    result['completed'] = index+1
    finally:
        pool.close()
        result['status'] = 'PASSED' if result.get('completed') == a.iterations and not result['mismatches'] else 'FAILED'
        (out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if 'sha256' not in k},indent=2))
    return 0 if result['status'] == 'PASSED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
