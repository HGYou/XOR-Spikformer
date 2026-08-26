#!/usr/bin/env python
"""Convert tiny-imagenet-200 into the ImageFolder layout timm expects.

The upstream archive ships

    train/<wnid>/images/*.JPEG          <- one extra 'images' level
    val/images/*.JPEG  +  val_annotations.txt

Both need to become

    train/<wnid>/*.JPEG
    val/<wnid>/*.JPEG

If train/ is left untouched, timm sees a single class directory named 'images',
every label becomes 0, and training silently collapses to chance accuracy while
the training loss still decreases. Always verify the class counts printed at
the end, and point --data-dir at the prepared directory.

usage:
    # write a prepared copy and leave the download untouched (default mode)
    python prepare_tiny_imagenet.py <src> --out <dst>

    # same, but hard-link the images instead of copying them
    python prepare_tiny_imagenet.py <src> --out <dst> --link

    # show what would be written, touching nothing
    python prepare_tiny_imagenet.py <src> --out <dst> --dry-run

    # rewrite the download itself; this moves files and cannot be undone
    python prepare_tiny_imagenet.py <src> --in-place
"""
import argparse
import os
import shutil
import sys

IMG_EXT = (".jpeg", ".jpg", ".png")


def _place(src, dst, mode, dry):
    """copy / hardlink / move one file, according to mode."""
    if dry:
        return 1
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return 1
    if mode == "move":
        shutil.move(src, dst)
    elif mode == "link":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)          # different volume, or no link support
    else:
        shutil.copy2(src, dst)
    return 1


def prepare_train(src, dst, mode, dry):
    """train/<wnid>/images/*.JPEG (or train/<wnid>/*.JPEG) -> <dst>/train/<wnid>/*.JPEG"""
    root = os.path.join(src, "train")
    if not os.path.isdir(root):
        sys.exit(f"no train/ under {src}")
    n = 0
    for wnid in sorted(os.listdir(root)):
        cls = os.path.join(root, wnid)
        if not os.path.isdir(cls):
            continue
        img = os.path.join(cls, "images")
        srcdir = img if os.path.isdir(img) else cls
        for f in sorted(os.listdir(srcdir)):
            if f.lower().endswith(IMG_EXT):
                n += _place(os.path.join(srcdir, f),
                            os.path.join(dst, "train", wnid, f), mode, dry)
        if mode == "move" and os.path.isdir(img) and not os.listdir(img):
            os.rmdir(img)
    return n


def prepare_val(src, dst, mode, dry):
    """val/images/*.JPEG + val_annotations.txt -> <dst>/val/<wnid>/*.JPEG"""
    root = os.path.join(src, "val")
    ann = os.path.join(root, "val_annotations.txt")
    img = os.path.join(root, "images")
    if not os.path.isfile(ann):
        sys.exit(f"no val/val_annotations.txt under {src}")
    srcdir = img if os.path.isdir(img) else root
    n = 0
    with open(ann) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            fname, wnid = parts[0], parts[1]
            s = os.path.join(srcdir, fname)
            if os.path.exists(s):
                n += _place(s, os.path.join(dst, "val", wnid, fname), mode, dry)
    if mode == "move" and os.path.isdir(img) and not os.listdir(img):
        os.rmdir(img)
    return n


def count(path):
    if not os.path.isdir(path):
        return 0, 0
    classes = [e for e in os.scandir(path) if e.is_dir()]
    files = sum(len([f for f in fs if f.lower().endswith(IMG_EXT)])
                for _, _, fs in os.walk(path))
    return len(classes), files


def main():
    ap = argparse.ArgumentParser(
        description="convert tiny-imagenet-200 to the ImageFolder layout",
        epilog="one of --out or --in-place is required")
    ap.add_argument("src", help="the unpacked tiny-imagenet-200 directory")
    ap.add_argument("--out", help="write the prepared dataset here; src is not modified")
    ap.add_argument("--link", action="store_true",
                    help="hard-link the images into --out instead of copying them")
    ap.add_argument("--in-place", action="store_true",
                    help="rewrite src itself by moving files; this cannot be undone")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be written and touch nothing")
    a = ap.parse_args()

    src = os.path.abspath(a.src)
    if not os.path.isdir(src):
        sys.exit(f"not a directory: {src}")
    if a.in_place == bool(a.out):
        sys.exit("pass either --out <dir> (safe) or --in-place (destructive), not both")

    if a.in_place:
        dst, mode = src, "move"
        if not a.dry_run:
            print(f"!! --in-place moves the images inside {src}; the original layout is lost")
    else:
        dst = os.path.abspath(a.out)
        mode = "link" if a.link else "copy"
        if os.path.abspath(dst) == src:
            sys.exit("--out must differ from src; use --in-place to rewrite src")

    print(f"src : {src}")
    print(f"dst : {dst}   ({mode}{', dry run' if a.dry_run else ''})")
    print(f"  train: {prepare_train(src, dst, mode, a.dry_run):6d} images")
    print(f"  val  : {prepare_val(src, dst, mode, a.dry_run):6d} images")

    if a.dry_run:
        print("\ndry run: nothing was written")
        return

    ok = True
    for split in ("train", "val"):
        c, n = count(os.path.join(dst, split))
        ok &= c == 200
        print(f"  {split:5s}: {c:3d} classes, {n:6d} files   {'OK' if c == 200 else 'FAIL'}")
    if not ok:
        sys.exit("\nboth splits must report 200 classes; training would collapse otherwise")
    print(f"\nready -- pass --data-dir {dst} to train.py")


if __name__ == "__main__":
    main()
