#!/usr/bin/env python3
import subprocess
import re
import shutil
import json
from typing import List, Dict

# Define the registries to support.
REGISTRIES = ['npm', 'pypi', 'maven', 'golang', 'ruby', 'hf', 'nuget']

# Mapping from registry to the command that prints its help text.
REGISTRY_HELP_COMMANDS = {
    'npm': ['npm', '--help'],
    'pypi': ['pip', '--help'],
    'maven': ['mvn', '--help'],
    'golang': ['go', 'help'],
    'ruby': ['gem', 'help', 'commands'],
    'hf': ['huggingface-cli', '--help'],
    'nuget': ['nuget', 'help']
}

def get_commands_from_help(help_text: str, registry: str = "") -> List[str]:
    """
    Parse the help text to extract command names.
    Uses specialized parsing for 'maven' and 'nuget', and default logic for others.
    """
    lines = help_text.splitlines()
    if registry == "maven":
        # Always return a fallback list of common Maven lifecycle phases.
        return ["clean", "validate", "compile", "test", "package", "verify", "install", "deploy"]
    elif registry == "nuget":
        commands = []
        in_commands_section = False
        empty_line_count = 0

        for line in lines:
            # Stop when we hit the footer
            if line.strip().startswith("For more information"):
                break

            if "Available commands:" in line:
                in_commands_section = True
                continue

            if in_commands_section:
                stripped = line.strip()

                # Handle empty lines
                if not stripped:
                    empty_line_count += 1
                    if empty_line_count > 1:
                        break
                    continue

                empty_line_count = 0  # Reset counter for non-empty lines

                # Extract command name (first word before any spaces or description)
                parts = stripped.split(None, 1)
                if parts:
                    # Remove any parentheses and question marks from command
                    command = parts[0].strip('()?')
                    # Filter out "NuGet's" which appears in command descriptions
                    if command and command != "NuGet's":
                        commands.append(command)

        if not commands:
            print(f"No commands found for registry '{registry}'")
        return commands
    else:
        # Default extraction: search for header keywords and split subsequent lines.
        commands = []
        header_pattern = re.compile(
            r'^\s*(Commands|Available Commands|All commands|The commands are|The available gem commands are):',
            re.IGNORECASE
        )
        collecting = False
        for line in lines:
            if not collecting:
                if header_pattern.search(line):
                    collecting = True
                continue
            else:
                if not line.strip():
                    continue
                # Stop collecting if an unindented line is reached and we have some commands.
                if line == line.lstrip() and commands:
                    break
                if ',' in line:
                    tokens = [token.strip() for token in line.split(',') if token.strip()]
                    commands.extend(tokens)
                else:
                    tokens = line.strip().split()
                    if tokens:
                        candidate = tokens[0].strip(',:')
                        if candidate:
                            commands.append(candidate)
        if not commands:
            for line in lines:
                m = re.match(r'^\s*([a-zA-Z][a-zA-Z0-9_-]+)\s{2,}', line)
                if m:
                    commands.append(m.group(1))
        return commands

def get_registry_commands(registry: str) -> List[str]:
    """
    Run the help command for the given registry and extract its built-in subcommands.
    Even if the help command returns a nonzero exit code, its output is captured and parsed.
    """
    help_cmd = REGISTRY_HELP_COMMANDS.get(registry)
    if not help_cmd:
        print(f"No help command defined for registry '{registry}'")
        return []
    if not shutil.which(help_cmd[0]):
        print(f"Command '{help_cmd[0]}' not found for registry '{registry}'")
        return []
    try:
        result = subprocess.run(
            help_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            check=False  # Allow nonzero exit codes.
        )
        output = result.stdout
        if result.returncode != 0:
            print(f"Warning: help command for {registry} exited with code {result.returncode}.")
    except Exception as e:
        print(f"Unexpected error for registry {registry}: {e}")
        return []

    commands = get_commands_from_help(output, registry)
    return commands

def main():
    registry_commands: Dict[str, List[str]] = {}
    print("Extracting built-in commands for each registry...\n")
    for registry in REGISTRIES:
        print(f"Processing registry: {registry}")
        cmds = get_registry_commands(registry)
        registry_commands[registry] = cmds
        print(f"  Found commands: {cmds}\n")

    # Save the registry commands to a JSON file.
    with open("./registry_commands.json", "w") as f:
        json.dump(registry_commands, f, indent=4)

    print("Final registry commands:")
    for registry, cmds in registry_commands.items():
        print(f"{registry}: {cmds}")

if __name__ == "__main__":
    main()
