. "$PSScriptRoot/ManageBuildNumbers.ps1"

Push-Location "$PSScriptRoot/.."
Update-BuildVersionFromRevision
dotnet pack -c Release
Pop-Location
