# L-Bot GitHub Migration Skeleton

This repo skeleton gives you:
- GitHub Actions CI that runs a quick backtest, prints metrics, and gates PRs by Sharpe/MaxDD.
- Helper scripts: `tasks.sh` and `setup_git.ps1`.

## Quick start
1. Create a **private** GitHub repo.
2. Copy all files in this skeleton to your project root.
3. (Windows) Run once to set remote with PAT:
   ```pwsh
   .\setup_git.ps1 -RepoUrl https://github.com/<you>/<repo>.git -Pat <YOUR_PAT>
   ```
4. Commit & push:
   ```bash
   git add -A && git commit -m "bootstrap" && git push
   ```
5. Open GitHub → Actions tab to see the CI run.

> Replace `backtests/run_quick_backtest.py` with your real backtest when ready.
