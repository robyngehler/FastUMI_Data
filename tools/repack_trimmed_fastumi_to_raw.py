#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import h5py


def natural_key(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', value)]


@dataclass
class EpisodeSpec:
    episode_index: int
    source_dir: Path
    source_hdf5: Path
    validation_errors: list[str] | None = None


@dataclass
class RunSpec:
    run_name: str
    run_dir: Path
    trimmed_dir: Path
    episodes: list[EpisodeSpec]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Repack trimmed FastUMI runs from test*_run*/trimmed/episode_*_<label>/episode_*.hdf5 '
            'into grouped raw directories such as pour_coke_v0/episode_0.hdf5.'
        )
    )
    parser.add_argument(
        '--dataset-root',
        type=Path,
        default=Path(__file__).resolve().parents[1] / 'dataset',
        help='Root containing recorded run folders such as test1_run01, test1_run02, ...',
    )
    parser.add_argument(
        '--output-root',
        type=Path,
        required=True,
        help='Directory where grouped raw folders (for example pour_coke_v0) will be created.',
    )
    parser.add_argument(
        '--run-glob',
        default='test*_run*',
        help='Glob used under --dataset-root to select run folders.',
    )
    parser.add_argument(
        '--trim-dir-name',
        default='trimmed',
        help='Name of the per-run directory that contains trimmed episode folders.',
    )
    parser.add_argument(
        '--trim-label',
        default='base_trim',
        help='Trim label suffix to collect, for example base_trim for episode_0_base_trim.',
    )
    parser.add_argument(
        '--group-prefix',
        default='pour_coke_v',
        help='Prefix used for generated output groups.',
    )
    parser.add_argument(
        '--expected-episodes',
        type=int,
        default=None,
        help='Optional fixed episode count required for every run, for example 3.',
    )
    parser.add_argument(
        '--skip-incomplete-runs',
        action='store_true',
        help='Skip runs whose trimmed episode count differs from the expected or dominant count.',
    )
    parser.add_argument(
        '--link-mode',
        choices=('symlink', 'hardlink', 'copy'),
        default='copy',
        help='How to materialize each trimmed HDF5 in the repacked output.',
    )
    parser.add_argument(
        '--validate-hdf5',
        action='store_true',
        help='Open each source HDF5 and validate action, observations/qpos, and observations/images/front before repacking.',
    )
    parser.add_argument(
        '--skip-corrupt-runs',
        action='store_true',
        help='Skip any run that contains at least one unreadable or structurally invalid HDF5 episode.',
    )
    parser.add_argument(
        '--manifest-name',
        default='repack_manifest.json',
        help='Manifest filename written under --output-root.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Remove an existing --output-root before writing new grouped folders.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Inspect and validate the structure without writing output files.',
    )
    return parser.parse_args()


def discover_episode_dirs(trimmed_dir: Path, trim_label: str) -> list[EpisodeSpec]:
    pattern = f'episode_*_{trim_label}'
    episode_specs: list[EpisodeSpec] = []
    for episode_dir in sorted(trimmed_dir.glob(pattern), key=lambda path: natural_key(path.name)):
        if not episode_dir.is_dir():
            continue
        match = re.match(rf'^episode_(\d+)_{re.escape(trim_label)}$', episode_dir.name)
        if match is None:
            continue
        episode_index = int(match.group(1))
        hdf5_candidates = sorted(episode_dir.glob('*.hdf5'), key=lambda path: natural_key(path.name))
        if not hdf5_candidates:
            raise FileNotFoundError(f'No .hdf5 file found in trimmed episode directory {episode_dir}')
        if len(hdf5_candidates) > 1:
            raise ValueError(f'Expected exactly one .hdf5 file in {episode_dir}, found {len(hdf5_candidates)}')
        episode_specs.append(
            EpisodeSpec(
                episode_index=episode_index,
                source_dir=episode_dir,
                source_hdf5=hdf5_candidates[0].resolve(),
            )
        )
    return sorted(episode_specs, key=lambda item: item.episode_index)


def discover_runs(dataset_root: Path, run_glob: str, trim_dir_name: str, trim_label: str) -> list[RunSpec]:
    run_specs: list[RunSpec] = []
    for run_dir in sorted(dataset_root.glob(run_glob), key=lambda path: natural_key(path.name)):
        if not run_dir.is_dir():
            continue
        trimmed_dir = run_dir / trim_dir_name
        if not trimmed_dir.is_dir():
            continue
        episodes = discover_episode_dirs(trimmed_dir, trim_label)
        if not episodes:
            continue
        run_specs.append(
            RunSpec(
                run_name=run_dir.name,
                run_dir=run_dir.resolve(),
                trimmed_dir=trimmed_dir.resolve(),
                episodes=episodes,
            )
        )
    if not run_specs:
        raise FileNotFoundError(
            f'No trimmed runs matching {run_glob!r} with label {trim_label!r} were found in {dataset_root}'
        )
    return run_specs


def validate_episode_hdf5(episode: EpisodeSpec) -> list[str]:
    errors: list[str] = []
    try:
        with h5py.File(episode.source_hdf5, 'r') as handle:
            for key in ('action', 'observations/qpos', 'observations/images/front'):
                try:
                    obj = handle[key]
                    _ = obj.shape
                except Exception as exc:
                    errors.append(f'{key}: {type(exc).__name__}: {exc}')
    except Exception as exc:
        errors.append(f'open: {type(exc).__name__}: {exc}')
    return errors


def validate_run_specs(run_specs: list[RunSpec], skip_corrupt_runs: bool) -> tuple[list[RunSpec], list[dict]]:
    valid_runs: list[RunSpec] = []
    skipped_runs: list[dict] = []
    for run_spec in run_specs:
        episode_failures = []
        for episode in run_spec.episodes:
            errors = validate_episode_hdf5(episode)
            episode.validation_errors = errors or None
            if errors:
                episode_failures.append(
                    {
                        'episode_index': episode.episode_index,
                        'source_hdf5': str(episode.source_hdf5),
                        'errors': errors,
                    }
                )
        if episode_failures:
            if skip_corrupt_runs:
                skipped_runs.append(
                    {
                        'run_name': run_spec.run_name,
                        'run_dir': str(run_spec.run_dir),
                        'episode_count': len(run_spec.episodes),
                        'reason': 'corrupt_hdf5',
                        'episodes': episode_failures,
                    }
                )
                continue
            raise ValueError(
                f'Corrupt or invalid HDF5 content detected in run {run_spec.run_name}: {episode_failures}'
            )
        valid_runs.append(run_spec)
    return valid_runs, skipped_runs


def filter_run_specs(
    run_specs: list[RunSpec],
    expected_episodes: int | None,
    skip_incomplete_runs: bool,
) -> tuple[list[RunSpec], list[RunSpec], int]:
    counts = [len(run_spec.episodes) for run_spec in run_specs]
    dominant_count = expected_episodes if expected_episodes is not None else max(counts)
    kept: list[RunSpec] = []
    skipped: list[RunSpec] = []
    for run_spec in run_specs:
        if len(run_spec.episodes) == dominant_count:
            kept.append(run_spec)
            continue
        if skip_incomplete_runs:
            skipped.append(run_spec)
            continue
        raise ValueError(
            'Inconsistent trimmed run sizes detected. '
            f'Expected {dominant_count} episode(s) per run, but {run_spec.run_name} has {len(run_spec.episodes)}. '
            'Use --skip-incomplete-runs to drop such runs or fix the missing trims first.'
        )
    if not kept:
        raise ValueError('No complete runs remain after applying the episode-count filter.')
    return kept, skipped, dominant_count


def prepare_output_root(output_root: Path, overwrite: bool):
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f'Output root already exists: {output_root}. Pass --overwrite to replace it.')
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def materialize_file(source: Path, destination: Path, link_mode: str):
    if link_mode == 'copy':
        shutil.copy2(source, destination)
        return
    if link_mode == 'hardlink':
        os.link(source, destination)
        return
    destination.symlink_to(source)


def build_manifest(
    kept_runs: list[RunSpec],
    skipped_runs: list[dict],
    dataset_root: Path,
    output_root: Path,
    group_prefix: str,
    trim_label: str,
    link_mode: str,
    dominant_count: int,
) -> dict:
    groups = []
    for group_index, run_spec in enumerate(kept_runs):
        groups.append(
            {
                'group_name': f'{group_prefix}{group_index}',
                'source_run': run_spec.run_name,
                'source_run_dir': str(run_spec.run_dir),
                'trimmed_dir': str(run_spec.trimmed_dir),
                'episode_count': len(run_spec.episodes),
                'episodes': [
                    {
                        'output_name': f'episode_{episode_offset}.hdf5',
                        'source_episode_index': episode.episode_index,
                        'source_hdf5': str(episode.source_hdf5),
                        'source_trimmed_dir': str(episode.source_dir),
                    }
                    for episode_offset, episode in enumerate(run_spec.episodes)
                ],
            }
        )

    return {
        'dataset_root': str(dataset_root),
        'output_root': str(output_root),
        'trim_label': trim_label,
        'link_mode': link_mode,
        'group_prefix': group_prefix,
        'episodes_per_group': dominant_count,
        'kept_run_count': len(kept_runs),
        'skipped_run_count': len(skipped_runs),
        'skipped_runs': [
            skipped_run for skipped_run in skipped_runs
        ],
        'groups': groups,
    }


def write_grouped_output(kept_runs: list[RunSpec], output_root: Path, group_prefix: str, link_mode: str):
    for group_index, run_spec in enumerate(kept_runs):
        group_dir = output_root / f'{group_prefix}{group_index}'
        group_dir.mkdir(parents=True, exist_ok=False)
        for episode_offset, episode in enumerate(run_spec.episodes):
            destination = group_dir / f'episode_{episode_offset}.hdf5'
            materialize_file(episode.source_hdf5, destination, link_mode)


def main():
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    run_specs = discover_runs(
        dataset_root=dataset_root,
        run_glob=args.run_glob,
        trim_dir_name=args.trim_dir_name,
        trim_label=args.trim_label,
    )

    skipped_runs: list[dict] = []
    if args.validate_hdf5:
        run_specs, corrupt_skipped_runs = validate_run_specs(
            run_specs=run_specs,
            skip_corrupt_runs=args.skip_corrupt_runs,
        )
        skipped_runs.extend(corrupt_skipped_runs)

    kept_runs, incomplete_run_specs, dominant_count = filter_run_specs(
        run_specs=run_specs,
        expected_episodes=args.expected_episodes,
        skip_incomplete_runs=args.skip_incomplete_runs,
    )
    skipped_runs.extend(
        {
            'run_name': run_spec.run_name,
            'run_dir': str(run_spec.run_dir),
            'episode_count': len(run_spec.episodes),
            'reason': 'incomplete_run',
        }
        for run_spec in incomplete_run_specs
    )

    manifest = build_manifest(
        kept_runs=kept_runs,
        skipped_runs=skipped_runs,
        dataset_root=dataset_root,
        output_root=output_root,
        group_prefix=args.group_prefix,
        trim_label=args.trim_label,
        link_mode=args.link_mode,
        dominant_count=dominant_count,
    )

    print(
        f'Found {len(kept_runs) + len(skipped_runs)} trimmed run(s); '
        f'keeping {len(kept_runs)} with {dominant_count} episode(s) each'
    )
    if skipped_runs:
        skipped_names = ', '.join(item['run_name'] for item in skipped_runs)
        print(f'Skipping incomplete runs: {skipped_names}')

    if args.dry_run:
        print('Dry run only. No files were written.')
        print(json.dumps(manifest, indent=2))
        return

    prepare_output_root(output_root, overwrite=args.overwrite)
    write_grouped_output(
        kept_runs=kept_runs,
        output_root=output_root,
        group_prefix=args.group_prefix,
        link_mode=args.link_mode,
    )

    manifest_path = output_root / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    print(f'Wrote grouped raw dataset to {output_root}')
    print(f'Wrote manifest to {manifest_path}')


if __name__ == '__main__':
    main()