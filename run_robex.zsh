if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    exit 1
fi

images_dir=$1
output_dir=$2

mkdir -p $output_dir
script="/home/fernando/opt/ROBEX/runROBEX.sh"
tempdir="/tmp/temp_robex"
mkdir -p $tempdir

for fp in `ls $images_dir`
do
    filename=`basename $fp`
    echo $filename
    output_path=${output_dir}/$filename
    if [ -f "$output_path" ]; then
        echo "$output_path exists. Skipping..."
    else
        $script $images_dir/$fp $tempdir/$filename $output_path
    fi
done
