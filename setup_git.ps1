param(
  [Parameter(Mandatory=$true)][string]$RepoUrl,   # https://github.com/<you>/<repo>.git
  [Parameter(Mandatory=$true)][string]$Pat        # Personal Access Token
)
git init
git remote remove origin 2>$null
$secureUrl = $RepoUrl -replace "^https://", "https://$Pat@"
git remote add origin $secureUrl
git add -A
if (-not (git rev-parse --verify HEAD 2>$null)) {
  git commit -m "bootstrap"
}
git branch -M main
git push -u origin main
Write-Host "✔ Git remote set. From now on: git push / pull works without retyping."
