# Modifications in advanced-minmpc

This distribution modifies upstream [PyCombina](https://github.com/adbuerger/pycombina),
version 0.3.4, commit `2a13c9e95f80fce2bb4e8dcc755a544cf951ebb1`.
The modified distribution and its changes are documented on **2026-10-31** by
the advanced-minmpc contributors. The original copyright notices remain intact.

PyCombina and these modifications are distributed under **LGPL-3.0-or-later**.
See [LICENSE.md](LICENSE.md) for the LGPL text, [COPYING](COPYING) for the
incorporated GPL text, and [ThirdPartyNotices.txt](ThirdPartyNotices.txt) for
additional notices. The repository's BSD-2-Clause license does not replace these terms.

Changes relative to the upstream commit:

- `setup.py`: native Windows build support using MinGW-w64 or MSVC.
- `pyproject.toml`: build and package discovery configuration; explicit inclusion of license and modification notices.
- `src/combina_bnb_solver/PythonBindings.cpp`: native-extension binding adaptations.
- `src/combina_bnb_solver/CombinaBnBSolver.cpp`: report failure when no feasible incumbent exists and allow feasible schedules at the full-horizon approximation bound.
- `src/pycombina/_binary_approximation.py`: retain zero-weight modes that combinatorial constraints may require.
- `src/pycombina/_combina_bnb.py`: propagate native solver failure instead of returning an invalid schedule.
- `.gitignore`: exclude local Python and native build artifacts.

Additional files provide Windows build instructions, this notice, the GPL text,
and source-distribution rules that include the C++ headers. Source code and build
instructions are provided so users can rebuild and replace the modified library.
