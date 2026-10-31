# pycombina on native Windows

This is a Windows-installable copy of upstream `pycombina` 0.3.4 at commit
`2a13c9e95f80fce2bb4e8dcc755a544cf951ebb1`, with Windows build adaptations and
fixes for constrained binary reconstruction. Preprocessing retains zero-weight
modes that constraints may require, and branch-and-bound reports infeasibility
instead of publishing an all-zero schedule. The initial bound also allows
feasible schedules whose approximation error equals the full horizon.

This modified distribution is documented on 2026-10-31. See
[MODIFICATIONS.md](MODIFICATIONS.md) for the changes and licensing notices.

Use 64-bit Python 3.8 or newer and either:

- a 64-bit MinGW-w64 toolchain (`g++.exe` and `dlltool.exe`), or
- Visual Studio 2022 Build Tools with **Desktop development with C++** and a
  Windows 10/11 SDK.

The build automatically detects MinGW on `PATH`, the MATLAB MinGW support
package, `C:\msys64\mingw64\bin`, or `C:\mingw64\bin`. For another location,
set `PYCOMBINA_MINGW_BIN` to its `bin` directory.

From the repository root in PowerShell, using the existing native Windows
virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .\pycombina_win
```

In VS Code, select `.venv\Scripts\python.exe` as the Python interpreter.

To run this repository's Lotka--Volterra MPC example:

```powershell
.\.venv\Scripts\python.exe -m pip install casadi matplotlib
.\.venv\Scripts\python.exe .\examples\lotka_volterra\lotka_volterra_pycombina_mpc.py
```

For the optional Gurobi-backed `CombinaMILP` solver, use:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".\pycombina_win[milp]"
```

Verify the native branch-and-bound solver:

```powershell
.\.venv\Scripts\python.exe -c "from pycombina import CombinaBnB; print('pycombina Windows OK')"
```

Run the upstream example (from the repository root):

```powershell
Set-Location .\pycombina_win\examples
..\..\.venv\Scripts\python.exe .\combina_example_1.py
```
