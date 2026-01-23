#!/usr/bin/env python3
"""Create a GIF from alpha_*.png files, sorted by file creation time."""

import argparse
from pathlib import Path
from PIL import Image


def get_creation_time(path: Path) -> float:
    """Get file creation time (or modification time as fallback)."""
    stat = path.stat()
    # On macOS, st_birthtime is the creation time
    # On Linux, fall back to st_mtime
    return getattr(stat, 'st_birthtime', stat.st_mtime)


def create_gif(
    input_dir: Path,
    output_path: Path,
    pattern: str = "alpha_*.png",
    duration: int = 100,
    loop: int = 0,
) -> None:
    """
    Create a GIF from PNG files matching the pattern.
    
    Args:
        input_dir: Directory containing the PNG files
        output_path: Path for the output GIF
        pattern: Glob pattern for input files
        duration: Duration of each frame in milliseconds
        loop: Number of loops (0 = infinite)
    """
    # Find all matching files
    png_files = list(input_dir.glob(pattern))
    
    if not png_files:
        raise ValueError(f"No files matching '{pattern}' found in {input_dir}")
    
    # Sort by creation time
    png_files.sort(key=get_creation_time)
    
    print(f"Found {len(png_files)} files")
    print(f"First: {png_files[0].name}")
    print(f"Last: {png_files[-1].name}")
    
    # Load images
    images = []
    for png_file in png_files:
        img = Image.open(png_file)
        # Convert to RGB if necessary (GIF doesn't support RGBA well)
        if img.mode == 'RGBA':
            # Create white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        images.append(img)
    
    # Save as GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
        optimize=True,
    )
    
    print(f"Created GIF: {output_path}")
    print(f"Frames: {len(images)}, Duration per frame: {duration}ms")


def main():
    parser = argparse.ArgumentParser(
        description="Create a GIF from alpha_*.png files sorted by creation time"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        nargs="?",
        default=Path("experiments/2026-01-22_bessel_radial_mix"),
        help="Directory containing PNG files (default: experiments/2026-01-22_bessel_radial_mix)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output GIF path (default: <input_dir>/alpha_animation.gif)",
    )
    parser.add_argument(
        "-p", "--pattern",
        type=str,
        default="alpha_*.png",
        help="Glob pattern for input files (default: alpha_*.png)",
    )
    parser.add_argument(
        "-d", "--duration",
        type=int,
        default=100,
        help="Duration of each frame in milliseconds (default: 100)",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=0,
        help="Number of loops, 0 for infinite (default: 0)",
    )
    
    args = parser.parse_args()
    
    input_dir = args.input_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")
    
    output_path = args.output or (input_dir / "alpha_animation.gif")
    
    create_gif(
        input_dir=input_dir,
        output_path=output_path,
        pattern=args.pattern,
        duration=args.duration,
        loop=args.loop,
    )


if __name__ == "__main__":
    main()
