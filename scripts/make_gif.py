#!/usr/bin/env python3
"""Create GIF and MP4 movies from PNG files, sorted by filename or creation time."""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


def get_creation_time(path: Path) -> float:
    """Get file creation time (or modification time as fallback)."""
    stat = path.stat()
    # On macOS, st_birthtime is the creation time
    # On Linux, fall back to st_mtime
    return getattr(stat, 'st_birthtime', stat.st_mtime)


def create_movie(
    input_dir: Path,
    output_base: Path = None,
    pattern: str = "*.png",
    duration: int = 100,
    loop: int = 0,
    sort_by: str = "name",
) -> Tuple[Path, Path]:
    """
    Create GIF and MP4 from PNG files matching the pattern.
    
    Args:
        input_dir: Directory containing the PNG files
        output_base: Base path for outputs (default: input_dir/movie)
                     Will create output_base.gif and output_base.mp4
        pattern: Glob pattern for input files
        duration: Duration of each frame in milliseconds
        loop: Number of loops (0 = infinite) - only applies to GIF
        sort_by: Sort method - "name" (alphabetical) or "time" (creation time)
    
    Returns:
        Tuple of (gif_path, mp4_path).
    """
    input_dir = Path(input_dir)
    if output_base is None:
        output_base = input_dir / "movie"
    else:
        output_base = Path(output_base)
        # Strip extension if provided
        if output_base.suffix in ('.gif', '.mp4'):
            output_base = output_base.with_suffix('')
    
    gif_path = output_base.with_suffix('.gif')
    mp4_path = output_base.with_suffix('.mp4')
    
    # Find all matching files
    png_files = list(input_dir.glob(pattern))
    
    if not png_files:
        raise ValueError(f"No files matching '{pattern}' found in {input_dir}")
    
    # Sort files
    if sort_by == "time":
        png_files.sort(key=get_creation_time)
    else:
        png_files.sort(key=lambda p: p.name)
    
    print(f"Found {len(png_files)} files")
    print(f"First: {png_files[0].name}")
    print(f"Last: {png_files[-1].name}")
    
    # Load images
    pil_images = []
    for png_file in tqdm(png_files, desc="Loading images"):
        img = Image.open(png_file)
        # Convert to RGB if necessary
        if img.mode == 'RGBA':
            # Create white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        pil_images.append(img)
    
    # For MP4: ensure all images are same size and divisible by 16 (macro_block_size)
    # Find the most common size and resize any outliers
    sizes = [img.size for img in pil_images]
    target_size = max(set(sizes), key=sizes.count)
    
    # Round up to nearest multiple of 16 for video compatibility
    target_w = ((target_size[0] + 15) // 16) * 16
    target_h = ((target_size[1] + 15) // 16) * 16
    mp4_size = (target_w, target_h)
    
    # Create numpy arrays with consistent size for MP4
    np_images = []
    for img in pil_images:
        if img.size != mp4_size:
            # Resize to target, padding with white if needed
            new_img = Image.new('RGB', mp4_size, (255, 255, 255))
            new_img.paste(img, (0, 0))
            np_images.append(np.array(new_img))
        else:
            np_images.append(np.array(img))
    
    # # Save as GIF
    # pil_images[0].save(
    #     gif_path,
    #     save_all=True,
    #     append_images=pil_images[1:],
    #     duration=duration,
    #     loop=loop,
    #     optimize=True,
    # )
    # print(f"Created GIF: {gif_path}")
    
    # Save as MP4 using imageio
    try:
        import imageio.v3 as iio
        fps = 1000 / duration  # Convert ms per frame to fps
        # Stack into video array and write
        video_array = np.stack(np_images, axis=0)
        iio.imwrite(mp4_path, video_array, fps=fps, codec='libx264')
        print(f"Created MP4: {mp4_path} ({mp4_size[0]}x{mp4_size[1]})")
    except ImportError:
        print("Warning: imageio not installed, skipping MP4 creation")
        print("  Install with: pip install imageio[ffmpeg]")
        mp4_path = None
    except Exception as e:
        print(f"Warning: Failed to create MP4: {e}")
        print("  You may need to install ffmpeg: pip install imageio[ffmpeg]")
        mp4_path = None
    
    print(f"Frames: {len(pil_images)}, Duration per frame: {duration}ms, FPS: {1000/duration:.1f}")
    
    return gif_path, mp4_path


# Alias for backward compatibility
def create_gif(input_dir, output_path=None, pattern="*.png", duration=100, loop=0, sort_by="name"):
    """Backward compatible wrapper - creates both GIF and MP4, returns GIF path."""
    gif_path, _ = create_movie(input_dir, output_path, pattern, duration, loop, sort_by)
    return gif_path


def main():
    parser = argparse.ArgumentParser(
        description="Create GIF and MP4 movies from PNG files in a directory"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing PNG files",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output base path (default: <input_dir>/movie). Creates .gif and .mp4",
    )
    parser.add_argument(
        "-p", "--pattern",
        type=str,
        default="*.png",
        help="Glob pattern for input files (default: *.png)",
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
        help="Number of loops, 0 for infinite (default: 0, GIF only)",
    )
    parser.add_argument(
        "--sort",
        type=str,
        choices=["name", "time"],
        default="name",
        help="Sort order: 'name' (alphabetical) or 'time' (creation time) (default: name)",
    )
    
    args = parser.parse_args()
    
    input_dir = args.input_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")
    
    create_movie(
        input_dir=input_dir,
        output_base=args.output,
        pattern=args.pattern,
        duration=args.duration,
        loop=args.loop,
        sort_by=args.sort,
    )


if __name__ == "__main__":
    main()
