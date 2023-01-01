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
Update-BuildVersionFromRevisionProjFile

$buildNumber = Get-BuildVersion
Echo "Building current version $buildNumber ..."

#$parametersTemplate = "/target:Rebuild /p:TargetFrameworkVersion={0};Configuration=Release;Platform={2};OutputPath=`"" + (Get-Location).Path + "`"\build\{1}\{3}\ /verbosity:Quiet"
$parametersTemplate = "/target:Rebuild /p:TargetFrameworkVersion={0};Configuration=Release;Platform={2};OutputPath=`"" + (Get-Location).Path + "`"\build\{1}\{3}\ "

#Echo "ParametersTemplate: " + ($parametersTemplate -f "v3.1", "netcoreapp3.1", "x64", "x64")

Invoke-MsBuild -Path "ManagedCUDA.sln" -MsBuildParameters ($parametersTemplate -f "v4.8", "net48", "NuGet_x64", "x64")
#Invoke-MsBuild -Path "ManagedCUDA.sln" -MsBuildParameters ($parametersTemplate -f "v4.5", "net45", "NuGet_x64", "x64")
#Invoke-MsBuild -Path "ManagedCUDA.sln" -MsBuildParameters ($parametersTemplate -f "v4.6", "net46", "NuGet_x64", "x64")
#Invoke-MsBuild -Path "ManagedCUDA.sln" -MsBuildParameters ($parametersTemplate -f "v4.7", "net47", "NuGet_x64", "x64")

dotnet build "ManagedCUDA.netCore.sln" -c Release -f netcoreapp3.1 --force --no-incremental -o build\netcoreapp3.1\x64\
dotnet build "ManagedCUDA.netCore.sln" -c Release -f net6.0 --force --no-incremental -o build\net6.0\x64\
dotnet build "ManagedCUDA.netCore.sln" -c Release -f net7.0 --force --no-incremental -o build\net7.0\x64\
Pop-Location
Pop-Location