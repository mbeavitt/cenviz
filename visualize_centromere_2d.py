#!/usr/bin/env python3
"""
Centromere-Wide 2D Repeat Visualization

Visualizes ALL repeats across a centromeric region as a 2D grid.
Each repeat monomer is shown as a colored tile, arranged in genomic order
(left-to-right, top-to-bottom). Colors represent sequence similarity via UMAP.

This gives a bird's-eye view of repeat patterns across the entire centromere,
ignoring HOR boundaries.

Usage:
    # Visualize ALL chromosomes (default behavior)
    python visualize_centromere_2d.py repeats.csv --name "ANGE-B10"

    # Visualize specific chromosome
    python visualize_centromere_2d.py repeats.csv --chromosome Chr6 \
        --name "ANGE-B10"

    # Visualize specific region of a chromosome
    python visualize_centromere_2d.py repeats.csv --chromosome Chr6 \
        --start 1000000 --end 8000000 --name "Sample-01"

    # Custom grid width
    python visualize_centromere_2d.py repeats.csv --chromosome Chr6 \
        --grid-width 100
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from umap import UMAP
import Levenshtein
import warnings
import pickle
import pandas as pd

from trash_compactor import utils


class CentromereVisualizer:
    """Visualize entire centromeric region as 2D grid of colored repeat tiles"""

    def __init__(self, repeats_table_path, chromosome, start=None, end=None,
                 output_dir='centromere_2d_output', name=None, precomputed_color_map=None,
                 filelist=None, cache_file=None):
        """
        Initialize the visualizer.

        Args:
            repeats_table_path: Path to TRASH repeats table CSV (or None if using filelist)
            chromosome: Chromosome name (e.g., 'Chr6')
            start: Start coordinate (bp), None for chromosome start
            end: End coordinate (bp), None for chromosome end
            output_dir: Output directory for figures
            name: Optional accession/sample name for title
            precomputed_color_map: Optional pre-computed global color map (for multi-chromosome runs)
            filelist: Optional path to file containing list of CSV paths (one per line)
            cache_file: Optional path to cache file for saving/loading color map
        """
        print("=" * 60)
        print("Centromere 2D Visualizer")
        print("=" * 60)

        self.chromosome = chromosome
        self.start = start
        self.end = end
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.cache_file = Path(cache_file) if cache_file else None

        # Load data from filelist or single file
        if filelist:
            print(f"\nLoading repeats from filelist: {filelist}")
            with open(filelist, 'r') as f:
                csv_paths = [line.strip() for line in f if line.strip()]

            print(f"  Found {len(csv_paths)} CSV files in filelist")

            # Load and combine all CSVs
            all_tables = []
            for csv_path in csv_paths:
                print(f"  Loading {csv_path}...")
                table = utils.import_repeats_table(csv_path)
                all_tables.append(table)
                print(f"    Loaded {len(table):,} repeats")

            self.full_repeats_table = pd.concat(all_tables, ignore_index=True)
            print(f"  Combined total: {len(self.full_repeats_table):,} repeats")
        else:
            self.repeats_table_path = Path(repeats_table_path)
            print(f"\nLoading repeats table from {repeats_table_path}")
            self.full_repeats_table = utils.import_repeats_table(repeats_table_path)
            print(f"  Loaded {len(self.full_repeats_table):,} total repeats")

        # Try to load cache, use precomputed, or compute new color map
        if precomputed_color_map is not None:
            print("\nUsing precomputed global color map...")
            self.global_color_map = precomputed_color_map
        elif self.cache_file and self.cache_file.exists():
            print(f"\nLoading color map from cache: {self.cache_file}")
            self.global_color_map = self._load_cache()
        else:
            print("\nComputing global color map from ALL chromosomes...")
            self._compute_global_colors()
            if self.cache_file:
                self._save_cache()

        # Filter to chromosome
        print(f"\nFiltering to {chromosome}...")
        self.repeats_table = self.full_repeats_table[
            self.full_repeats_table['seq_name'] == chromosome
        ].copy()

        if len(self.repeats_table) == 0:
            available = self.full_repeats_table['seq_name'].unique()
            raise ValueError(f"No repeats found for {chromosome}. Available: {available}")

        print(f"  Found {len(self.repeats_table):,} repeats on {chromosome}")

        # Filter to coordinate range if specified
        if start is not None or end is not None:
            original_count = len(self.repeats_table)
            if start is not None:
                self.repeats_table = self.repeats_table[
                    self.repeats_table['end'] > start
                ]
            if end is not None:
                self.repeats_table = self.repeats_table[
                    self.repeats_table['start'] < end
                ]
            print(f"  Filtered to region {start or 0:,} - {end or 'end':,}: "
                  f"{len(self.repeats_table):,} repeats")

        # Sort by genomic position
        self.repeats_table = self.repeats_table.sort_values('start').reset_index(drop=True)
        print(f"  Sorted by genomic position")

        # Store coordinate range
        self.actual_start = self.repeats_table['start'].min()
        self.actual_end = self.repeats_table['end'].max()
        print(f"  Actual range: {self.actual_start:,} - {self.actual_end:,} bp "
              f"({self.actual_end - self.actual_start:,} bp span)")

        print("\n" + "=" * 60)
        print("Initialization complete!")
        print("=" * 60 + "\n")

    def _compute_global_colors(self):
        """Compute global color mapping for all unique repeat sequences across ALL chromosomes"""

        # Get all unique sequences from ALL chromosomes
        print("  Extracting unique sequences from all chromosomes...")
        self.unique_sequences = sorted(self.full_repeats_table['sequence'].unique())
        n_unique = len(self.unique_sequences)

        print(f"  Found {n_unique:,} unique repeat sequences")

        # Build distance matrix
        print(f"  Building distance matrix ({n_unique:,} x {n_unique:,})...")
        distance_matrix = np.zeros((n_unique, n_unique))

        for i in range(n_unique):
            for j in range(i+1, n_unique):
                dist = Levenshtein.distance(self.unique_sequences[i], self.unique_sequences[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"    Computed {i + 1:,}/{n_unique:,} rows...")

        # Project to 3D RGB space using UMAP
        print(f"  Projecting to 3D color space using UMAP...")
        warnings.filterwarnings('ignore', message='using precomputed metric')
        warnings.filterwarnings('ignore', message='n_jobs value')
        warnings.filterwarnings('ignore', message='Graph is not fully connected')

        reducer = UMAP(
            n_components=3,
            metric='precomputed',
            n_neighbors=min(15, n_unique - 1),
            min_dist=0.1,
            n_jobs=1,
            init='random',
            random_state=42
        )
        rgb_coords = reducer.fit_transform(distance_matrix)

        # Normalize to [0, 1] for valid RGB values
        scaler = MinMaxScaler()
        rgb_coords = scaler.fit_transform(rgb_coords)
        rgb_coords = np.clip(rgb_coords, 0, 1)

        # Create color map
        self.global_color_map = {seq: rgb_coords[i] for i, seq in enumerate(self.unique_sequences)}

        print(f"  Color map computed for {n_unique:,} unique sequences")

        # Save 3D UMAP visualization
        self._save_3d_color_space(rgb_coords)

    def _save_cache(self):
        """Save the global color map to cache file"""
        if self.cache_file:
            print(f"\nSaving color map to cache: {self.cache_file}")
            cache_data = {
                'global_color_map': self.global_color_map,
                'unique_sequences': self.unique_sequences
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"  Cache saved ({len(self.global_color_map):,} sequences)")

    def _load_cache(self):
        """Load the global color map from cache file"""
        with open(self.cache_file, 'rb') as f:
            cache_data = pickle.load(f)

        self.unique_sequences = cache_data['unique_sequences']
        print(f"  Loaded color map for {len(cache_data['global_color_map']):,} unique sequences")
        return cache_data['global_color_map']

    def _save_3d_color_space(self, rgb_coords):
        """Save 3D visualization of the color space"""
        from mpl_toolkits.mplot3d import Axes3D

        print(f"  Creating 3D UMAP visualization...")
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(
            rgb_coords[:, 0], rgb_coords[:, 1], rgb_coords[:, 2],
            c=rgb_coords, s=50, alpha=0.8, edgecolors='k', linewidth=0.5
        )

        ax.set_xlabel('Red (UMAP 1)')
        ax.set_ylabel('Green (UMAP 2)')
        ax.set_zlabel('Blue (UMAP 3)')
        ax.set_title(f'3D UMAP Color Space: {len(self.unique_sequences):,} Unique Sequences\n'
                     f'(Points colored by their RGB values)')

        output_path = self.output_dir / 'umap_3d_color_space.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved 3D UMAP plot to {output_path}")

    def _calculate_grid_width(self, n_repeats, target_aspect_ratio=2.0):
        """
        Calculate optimal grid width for wide rectangular layout.

        Args:
            n_repeats: Total number of repeats
            target_aspect_ratio: Desired width/height ratio (default 2:1 for wide view)

        Returns:
            Grid width (repeats per row)
        """
        optimal_width = int(np.sqrt(n_repeats * target_aspect_ratio))

        # Clamp to reasonable bounds
        min_width = 50   # Minimum width for patterns to be visible
        max_width = 200  # Maximum for reasonable resolution

        return np.clip(optimal_width, min_width, max_width)

    def visualize(self, grid_width='auto', tile_size=10, output_name='centromere_2d.png'):
        """
        Create 2D grid visualization of entire centromeric region.

        Args:
            grid_width: 'auto' or integer (repeats per row)
            tile_size: Pixels per repeat tile
            output_name: Output filename

        Returns:
            Path to saved figure
        """
        n_repeats = len(self.repeats_table)

        print(f"Creating 2D grid visualization...")
        print(f"  Total repeats: {n_repeats:,}")

        # Calculate grid width
        if grid_width == 'auto':
            grid_width = self._calculate_grid_width(n_repeats)
            print(f"  Auto-calculated grid width: {grid_width} repeats/row")
        else:
            grid_width = int(grid_width)
            print(f"  Using grid width: {grid_width} repeats/row")

        n_rows = int(np.ceil(n_repeats / grid_width))
        print(f"  Grid dimensions: {grid_width} x {n_rows} = {grid_width * n_rows:,} tiles")

        # Build 2D RGB array
        print(f"  Building RGB array...")
        height_px = n_rows * tile_size
        width_px = grid_width * tile_size

        rgb_array = np.ones((height_px, width_px, 3)) * 0.8  # Gray background

        # Fill in each repeat tile
        sequences = self.repeats_table['sequence'].values
        for idx, seq in enumerate(sequences):
            row = idx // grid_width
            col = idx % grid_width

            y_start = row * tile_size
            y_end = (row + 1) * tile_size
            x_start = col * tile_size
            x_end = (col + 1) * tile_size

            if seq in self.global_color_map:
                rgb_array[y_start:y_end, x_start:x_end] = self.global_color_map[seq]

            # Progress indicator
            if (idx + 1) % 10000 == 0:
                print(f"    Filled {idx + 1:,}/{n_repeats:,} tiles...")

        # Create figure
        print(f"  Creating figure...")
        fig_width = min(30, max(12, grid_width * tile_size / 100))
        fig_height = min(40, max(8, n_rows * tile_size / 100))
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        ax.imshow(rgb_array, aspect='auto', interpolation='nearest')
        ax.axis('off')

        # Add title with metadata
        name_str = f'{self.name} - ' if self.name else ''
        title = (f'Centromere 2D Visualization: {name_str}{self.chromosome}\n'
                 f'Region: {self.actual_start:,} - {self.actual_end:,} bp '
                 f'({(self.actual_end - self.actual_start)/1e6:.2f} Mb)\n'
                 f'{n_repeats:,} repeats arranged in {grid_width} columns x {n_rows} rows')
        plt.suptitle(title, fontsize=14, y=0.98)

        # Add grid lines every N rows for easier reading
        if n_rows > 10:
            grid_interval = max(10, n_rows // 20)  # ~20 grid lines max
            for i in range(0, n_rows, grid_interval):
                y_pos = i * tile_size
                ax.axhline(y=y_pos, color='white', linewidth=0.5, alpha=0.3)

        plt.tight_layout()

        # Save
        output_path = self.output_dir / output_name
        print(f"  Saving figure...")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n✓ Saved: {output_path}")
        print(f"  Image size: {width_px} x {height_px} pixels")

        return output_path

    def visualize_with_annotations(self, grid_width='auto', tile_size=10,
                                   annotation_interval=100000):
        """
        Create annotated visualization with genomic position markers.

        Args:
            grid_width: 'auto' or integer
            tile_size: Pixels per tile
            annotation_interval: Show position marker every N bp

        Returns:
            Path to saved figure
        """
        n_repeats = len(self.repeats_table)

        # Calculate grid width
        if grid_width == 'auto':
            grid_width = self._calculate_grid_width(n_repeats)

        print(f"Creating annotated 2D visualization...")
        print(f"  Grid: {grid_width} repeats/row")

        # Build RGB array (same as visualize())
        n_rows = int(np.ceil(n_repeats / grid_width))
        height_px = n_rows * tile_size
        width_px = grid_width * tile_size
        rgb_array = np.ones((height_px, width_px, 3)) * 0.8

        sequences = self.repeats_table['sequence'].values
        for idx, seq in enumerate(sequences):
            row = idx // grid_width
            col = idx % grid_width
            y_start = row * tile_size
            y_end = (row + 1) * tile_size
            x_start = col * tile_size
            x_end = (col + 1) * tile_size

            if seq in self.global_color_map:
                rgb_array[y_start:y_end, x_start:x_end] = self.global_color_map[seq]

        # Create figure with annotations
        fig, ax = plt.subplots(figsize=(20, max(10, n_rows * tile_size / 100)))
        ax.imshow(rgb_array, aspect='auto', interpolation='nearest')

        # Add position annotations
        positions = self.repeats_table['start'].values
        for idx, pos in enumerate(positions):
            if pos % annotation_interval < 200:  # Near annotation boundary
                row = idx // grid_width
                col = idx % grid_width
                y_pos = row * tile_size

                # Add marker line
                ax.axhline(y=y_pos, color='yellow', linewidth=1, alpha=0.5)

                # Add text label
                ax.text(0, y_pos, f'{pos/1e6:.2f} Mb',
                       fontsize=8, color='yellow', va='bottom',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        ax.axis('off')

        name_str = f'{self.name} - ' if self.name else ''
        title = (f'Annotated Centromere View: {name_str}{self.chromosome}\n'
                 f'{n_repeats:,} repeats from {self.actual_start:,} to {self.actual_end:,} bp')
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()

        output_path = self.output_dir / 'centromere_2d_annotated.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved annotated: {output_path}")
        return output_path


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Visualize entire centromeric region as 2D repeat grid',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize ALL chromosomes (default)
  %(prog)s repeats.csv --name "ANGE-B10"

  # Visualize specific chromosome
  %(prog)s repeats.csv --chromosome Chr6 --name "ANGE-B10"

  # Visualize specific region
  %(prog)s repeats.csv --chromosome Chr6 --start 1000000 --end 8000000

  # Custom grid layout
  %(prog)s repeats.csv --chromosome Chr6 --grid-width 100 --tile-size 15

  # With position annotations
  %(prog)s repeats.csv --chromosome Chr6 --annotated --name "Sample-01"
        """
    )

    parser.add_argument('repeats_table', nargs='?',
                       help='Path to TRASH repeats table CSV')
    parser.add_argument('--filelist',
                       help='Path to file containing list of CSV paths (one per line)')
    parser.add_argument('--cache-file',
                       help='Path to cache file for saving/loading UMAP color map')
    parser.add_argument('--chromosome',
                       help='Chromosome name (e.g., Chr6). If not specified, plots ALL chromosomes.')
    parser.add_argument('--start', type=int,
                       help='Start coordinate in bp (optional)')
    parser.add_argument('--end', type=int,
                       help='End coordinate in bp (optional)')
    parser.add_argument('-o', '--output-dir', default='centromere_2d_output',
                       help='Output directory (default: centromere_2d_output)')
    parser.add_argument('--grid-width', default='auto',
                       help='Repeats per row: "auto" or integer (default: auto)')
    parser.add_argument('--tile-size', type=int, default=10,
                       help='Pixels per tile (default: 10)')
    parser.add_argument('--annotated', action='store_true',
                       help='Create annotated version with position markers')
    parser.add_argument('--output-name', default='centromere_2d.png',
                       help='Output filename (default: centromere_2d.png)')
    parser.add_argument('--name',
                       help='Accession/sample name to display in title (e.g., "ANGE-B10")')

    args = parser.parse_args()

    # Validate input
    if not args.repeats_table and not args.filelist:
        parser.error("Either repeats_table or --filelist must be provided")

    # Load repeats table to get available chromosomes
    if args.filelist:
        print(f"\nLoading repeats from filelist: {args.filelist}")
        with open(args.filelist, 'r') as f:
            csv_paths = [line.strip() for line in f if line.strip()]

        print(f"  Found {len(csv_paths)} CSV files")

        # Load first CSV to get chromosomes
        full_repeats_table = utils.import_repeats_table(csv_paths[0])
        print(f"  Loaded first CSV: {len(full_repeats_table):,} repeats")
    else:
        print(f"\nLoading repeats table from {args.repeats_table}")
        full_repeats_table = utils.import_repeats_table(args.repeats_table)
        print(f"  Loaded {len(full_repeats_table):,} total repeats")

    # Determine which chromosomes to process
    if args.chromosome:
        chromosomes = [args.chromosome]
    else:
        chromosomes = sorted(full_repeats_table['seq_name'].unique())
        print(f"\nNo chromosome specified - will plot ALL {len(chromosomes)} chromosomes")

    # Process first chromosome (or single specified chromosome) to build global color map
    first_chrom = chromosomes[0]
    viz = CentromereVisualizer(
        args.repeats_table,
        first_chrom,
        start=args.start if args.chromosome else None,
        end=args.end if args.chromosome else None,
        output_dir=args.output_dir,
        name=args.name,
        filelist=args.filelist,
        cache_file=args.cache_file
    )

    # Create visualization for first chromosome
    output_name = args.output_name if args.chromosome else f'centromere_2d_{first_chrom}.png'
    viz.visualize(
        grid_width=args.grid_width,
        tile_size=args.tile_size,
        output_name=output_name
    )

    if args.annotated:
        viz.visualize_with_annotations(
            grid_width=args.grid_width,
            tile_size=args.tile_size
        )

    # Process remaining chromosomes if plotting all
    if not args.chromosome and len(chromosomes) > 1:
        # Reuse the global color map from first visualizer
        global_color_map = viz.global_color_map

        for chrom in chromosomes[1:]:
            print(f"\n{'=' * 60}")
            print(f"Processing {chrom}...")
            print("=" * 60)

            # Create new visualizer with shared global color map
            viz_chrom = CentromereVisualizer(
                args.repeats_table,
                chrom,
                start=None,
                end=None,
                output_dir=args.output_dir,
                name=args.name,
                precomputed_color_map=global_color_map,
                filelist=args.filelist,
                cache_file=None  # Only save cache once from first visualizer
            )

            viz_chrom.visualize(
                grid_width=args.grid_width,
                tile_size=args.tile_size,
                output_name=f'centromere_2d_{chrom}.png'
            )

            if args.annotated:
                viz_chrom.visualize_with_annotations(
                    grid_width=args.grid_width,
                    tile_size=args.tile_size
                )

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
