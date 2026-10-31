#!/usr/bin/python
# -*- coding: utf-8 -*-
# Modified for advanced-minmpc; changes documented 2026-10-31 in MODIFICATIONS.md.
#
# This file is part of pycombina.
#
# Copyright 2017-2018 Adrian Bürger, Clemens Zeile, Sebastian Sager, Moritz Diehl
#
# pycombina is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pycombina is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with pycombina. If not, see <http://www.gnu.org/licenses/>.

import os
import shutil
import sys
from pathlib import Path

import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


SOURCE_ROOT = Path("src") / "combina_bnb_solver"


def _find_mingw_bin():
    """Find a 64-bit MinGW toolchain already available on Windows."""
    configured = os.environ.get("PYCOMBINA_MINGW_BIN")
    if configured:
        candidates = [Path(configured)]
    else:
        candidates = []
        compiler = shutil.which("g++")
        if compiler:
            candidates.append(Path(compiler).parent)

        program_data = Path(os.environ.get("ProgramData", r"C:\ProgramData"))
        candidates.extend(
            sorted(
                program_data.glob(
                    "MATLAB/SupportPackages/R*/3P.instrset/"
                    "mingw_w64.instrset/bin"
                ),
                reverse=True,
            )
        )
        candidates.extend(
            [
                Path(r"C:\msys64\mingw64\bin"),
                Path(r"C:\mingw64\bin"),
            ]
        )

    for candidate in candidates:
        if (candidate / "g++.exe").is_file() and (
            candidate / "dlltool.exe"
        ).is_file():
            return candidate.resolve()
    return None


class WindowsCompatibleBuildExt(build_ext):
    """Use MinGW when present and otherwise retain setuptools' MSVC path."""

    def finalize_options(self):
        super().finalize_options()
        if sys.platform == "win32" and self.compiler is None:
            mingw_bin = _find_mingw_bin()
            if mingw_bin is not None:
                os.environ["PATH"] = os.pathsep.join(
                    (str(mingw_bin), os.environ.get("PATH", ""))
                )
                self.compiler = "mingw32"

    def build_extensions(self):
        if self.compiler.compiler_type == "mingw32":
            for extension in self.extensions:
                extension.extra_compile_args = ["-std=c++14", "-DNDEBUG"]
                extension.extra_link_args = [
                    "-static",
                    "-static-libgcc",
                    "-static-libstdc++",
                ]
        super().build_extensions()

# Set up extension module.
setup(
    cmdclass={"build_ext": WindowsCompatibleBuildExt},
    ext_modules=[
        Extension(
            name="pycombina._combina_bnb_solver",
            sources=[str(path) for path in SOURCE_ROOT.rglob("*.cpp")],
            include_dirs=[pybind11.get_include()],
            language="c++",
        ),
    ]
)
