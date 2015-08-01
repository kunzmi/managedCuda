function Update-BuildVersionFromRevision
{  
    [CmdletBinding()]  
    param()  
          
    $assemblyPattern = "[0-9]+(\.([0-9]+|\*)){1,3}"  
    $assemblyVersionPattern = 'AssemblyVersion\("([0-9]+(\.([0-9]+|\*)){1,3})"\)'  
      
    $foundFiles = get-childitem .\*GlobalAssemblyInfo.cs -recurse  
                         
              
    $rawVersionNumberGroup = get-content $foundFiles | select-string -pattern $assemblyVersionPattern | select -first 1 | % { $_.Matches }              
    $rawVersionNumber = $rawVersionNumberGroup.Groups[1].Value  
                    
    $hgCommitDistance = hg log -r tip --template '{rev}'

    $versionParts = $rawVersionNumber.Split('.')       
    $updatedAssemblyVersion = "{0}.{1}.{2}" -f $versionParts[0], $versionParts[1], $hgCommitDistance 
      
    $assemblyVersion  
                  
    foreach( $file in $foundFiles )  
    {     
        (Get-Content $file) | ForEach-Object {
                % {$_ -replace $assemblyPattern, $updatedAssemblyVersion }                 
            } | Set-Content $file        
    }  
}

function Get-BuildVersion  
{  
    [CmdletBinding()]  
    param()  
          
    $assemblyPattern = "[0-9]+(\.([0-9]+|\*)){1,3}"  
    $assemblyVersionPattern = 'AssemblyVersion\("([0-9]+(\.([0-9]+|\*)){1,3})"\)'  
      
    $foundFiles = get-childitem .\*GlobalAssemblyInfo.cs -recurse  
                         
              
    $rawVersionNumberGroup = get-content $foundFiles | select-string -pattern $assemblyVersionPattern | select -first 1 | % { $_.Matches }              
    $rawVersionNumber = $rawVersionNumberGroup.Groups[1].Value  
                    
    $hgCommitDistance = hg log -r tip --template '{rev}'

    $versionParts = $rawVersionNumber.Split('.')       
    $updatedAssemblyVersion = "{0}.{1}.{2}" -f $versionParts[0], $versionParts[1], $hgCommitDistance 
      
    return $updatedAssemblyVersion
} 