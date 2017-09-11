function Update-BuildVersionFromRevision
{  
    [CmdletBinding()]  
    param()  
          
    $gitCommitDistance = git rev-list HEAD --count
    Set-Content -Path "Revision.targets" -Value "<Project><PropertyGroup><Rev>$gitCommitDistance</Rev></PropertyGroup></Project>" 
}

