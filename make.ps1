
$currentPath = $pwd

if($args.length -gt 0) {
    $profile=$args[0]
}

Invoke-Expression -Command:"mvn -f pom.xml clean package -U -DskipTests"

$projs=@("java_audio_classifier", "java_audio_melgram")
foreach ($proj in $projs){
    $source=$PSScriptRoot + "/" + $proj + "/target/" + $proj + ".jar"
    $dest=$PSScriptRoot + "/" + $proj + ".jar"
    copy $source $dest
}

cd $currentPath
