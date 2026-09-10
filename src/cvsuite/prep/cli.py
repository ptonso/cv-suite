def main(argv: list[str] | None = None) -> int:
    import argparse
    import importlib
    import os
    import pkgutil
    from pathlib import Path

    package = __package__ or "cvsuite.prep"
    prog = "cvsuite prep"
    pkg = importlib.import_module(package)
    root = Path(pkg.__file__).resolve().parent

    parser = argparse.ArgumentParser(prog=prog, description="Vision dataset cleaning CLI.")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)
    commands_dirs = []
    for dirpath, dirnames, _filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in {"venv", "__pycache__"}]
        current = Path(dirpath)
        if current.name == "commands":
            commands_dirs.append(current)

    for commands_dir in sorted(commands_dirs):
        rel_parts = commands_dir.relative_to(root).parts
        import_base = ".".join([package, *rel_parts])
        for info in pkgutil.iter_modules([str(commands_dir)]):
            if info.ispkg:
                continue
            module = importlib.import_module(f"{import_base}.{info.name}")
            register = getattr(module, "register_subcommand", None)
            if register is not None:
                register(subparsers)

    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.error("No command selected.")
    result = func(args)
    return 0 if result is None else int(result)


if __name__ == "__main__":
    raise SystemExit(main())
