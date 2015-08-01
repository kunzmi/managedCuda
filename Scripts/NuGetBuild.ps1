Push-Location $PSScriptRoot

$currentLocation = Get-Location
Import-Module -Name ( $currentLocation.Path + "/Invoke-MsBuild.psm1" )
. ".\ManageBuildNumbers.ps1"

# Move to the parent (SolutionDir).
Push-Location ..

###$nugetExecutable = Get-ChildItem "NuGet.exe" -Recurse | Select -First 1 
$nugetExecutable = "NuGet.exe" 
Set-Alias Execute-NuGet $nugetExecutable

Update-BuildVersionFromRevision

$buildNumber = Get-BuildVersion
Echo "Building current version $buildNumber ..."

$parametersTemplate = "/target:Rebuild /p:TargetFrameworkVersion={0};Configuration=Release;Platform={2};OutputPath=`"" + (Get-Location).Path + "`"\build\{1}\{3}\ /verbosity:Quiet"

#Echo "ParametersTemplate: " + ($parametersTemplate -f "v4.0", "net40", "NuGet_x86")

Invoke-MsBuild -Path "ManagedCUDA.sln" -MsBuildParameters ($parametersTemplate -f "v4.0", "net40", "NuGet_x86", "x86")
Invoke-MsBuild -Path "ManagedCUDA.sln" -MsBuildParameters ($parametersTemplate -f "v4.0", "net40", "NuGet_x64", "x64")
Invoke-MsBuild -Path "ManagedCUDA.sln" -MsBuildParameters ($parametersTemplate -f "v4.5", "net45", "NuGet_x86", "x86")
Invoke-MsBuild -Path "ManagedCUDA.sln" -MsBuildParameters ($parametersTemplate -f "v4.5", "net45", "NuGet_x64", "x64")


Pop-Location
Pop-Location