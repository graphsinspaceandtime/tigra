$NETREP_URLS = @(
    "http://nrvis.com/download/data/misc/radoslaw_email.zip"
)

$NETREP_DIRS = @(
    "radoslaw_email"
)

$NETREP_CHECK_FILES = @(
    "radoslaw_email.edges"
)

$NETREP_CHECKSUMS = @(
    "cbdc821eb8949cfa36fb71b14aa3cf745e977de1fd2de0ece099c113df364969"
)

$SNAP_URLS = @(
    "https://snap.stanford.edu/data/CollegeMsg.txt.gz",
    "https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz",
    "https://snap.stanford.edu/data/sx-askubuntu.txt.gz",
    "https://snap.stanford.edu/data/sx-mathoverflow.txt.gz",
    "https://snap.stanford.edu/data/sx-stackoverflow.txt.gz",
    "https://snap.stanford.edu/data/sx-superuser.txt.gz",
    "https://snap.stanford.edu/data/wiki-talk-temporal.txt.gz"
)

$SNAP_FILENAMES = @(
    "CollegeMsg.txt.gz",
    "email-Eu-core-temporal.txt.gz",
    "sx-askubuntu.txt.gz",
    "sx-mathoverflow.txt.gz",
    "sx-stackoverflow.txt.gz",
    "sx-superuser.txt.gz",
    "wiki-talk-temporal.txt.gz"
)

$SNAP_CHECKSUMS = @(
    "e00ba2415373dee52c00616065bcceaa4750e78de60d1855c76470600f10740f",
    "1835c94dc6685dcf2a614a1b6a0275b323dd4002f7aa111fbe9f76db8864391f",
    "1a416ccaf5a679996156dd31290dfcca3e7dde3431171afed1b1fd616fbd31e6",
    "2bf52f784cdba69730f912633aa1414b381059406ded2c38bc37658ead5b0773",
    "22a717fd1763199c95b10dbea80af4fb658f0b9e71cd9eb8a1fff51462238cdc",
    "d99a66429d0cb69b6f61be71cac3064868cd683b251140b232f377bf42112be7",
    "d45012e758e26bc28f6b610f1146aee0e87728a79e69cc6d696dd180f19dff66"
)

foreach ($i in 0..($NETREP_URLS.Length - 1)) {
    $checksum = Get-FileHash -Path "netrep\$($NETREP_DIRS[$i])\$($NETREP_CHECK_FILES[$i])" -Algorithm SHA256 -ErrorAction SilentlyContinue
    if (-not $checksum) {
        Write-Host "Checksum incorrect or file not found for $($NETREP_DIRS[$i])"
        Write-Host "Downloading $($NETREP_DIRS[$i])"
        Remove-Item -Path "netrep\$($NETREP_DIRS[$i])" -Recurse -Force
        $tempFileName = [System.IO.Path]::GetTempFileName()
        Invoke-WebRequest -Uri $NETREP_URLS[$i] -OutFile $tempFileName
        tar -xvzf $tempFileName -C "netrep\$($NETREP_DIRS[$i])" -Force
        Remove-Item -Path $tempFileName -Force
    } else {
        Write-Host "Skipping $($NETREP_DIRS[$i])... already exists"
    }
}

mkdir "snap" -ErrorAction SilentlyContinue
foreach ($i in 0..($SNAP_URLS.Length - 1)) {
    $checksum = Get-FileHash -Path "snap\$($SNAP_FILENAMES[$i].Replace('.gz', ''))" -Algorithm SHA256 -ErrorAction SilentlyContinue
    if (-not $checksum) {
        Write-Host "Checksum incorrect or file not found for $($SNAP_FILENAMES[$i])"
        Write-Host "Downloading $($SNAP_FILENAMES[$i].Replace('.gz', ''))"
        Remove-Item -Path "snap\$($SNAP_FILENAMES[$i])" -Force
        Invoke-WebRequest -Uri $SNAP_URLS[$i] -OutFile "snap\$($SNAP_FILENAMES[$i])"
        tar -xvzf "snap\$($SNAP_FILENAMES[$i])" -C "snap" -Force
    } else {
        Write-Host "Skipping $($SNAP_FILENAMES[$i].Replace('.gz', ''))... already exists"
    }
}
