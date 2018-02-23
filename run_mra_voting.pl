#!/usr/bin/perl

# runs the raw data , wavelet detail/approx matching voting scheme
my @data_files = ("samples_unt.txt", "samples_wavelet1ap.txt", "samples_wavelet1dl.txt", 
 "samples_wavelet2ap.txt", "samples_wavelet2dl.txt", 
"samples_wavelet3ap.txt", "samples_wavelet3dl.txt",
"samples_wavelet4ap.txt", "samples_wavelet4dl.txt");

my $nrecords = 0;
open(FILE,"data/$data_files[0]") or die "can't open $!";
$nrecords++ while <FILE>;
close FILE;
$nrecords--; # header

my @record_nums = (0 .. ($nrecords-1));
my $ntest = $ARGV[0]; # of test samples
my $ntrain = $nrecords - $ntest;
my $cookie = 1001;
print("using ntest = $ntest\n");
print("nrecords = $nrecords; run with ntrain=$ntrain, ntest=$ntest\n");


my $record_fname = "data/record_nums.txt";
my $test_labels_fname = "data/test_labels1.txt";
my $train_labels_fname = "data/train_labels1.txt";
my $predicted_labels_fname = "data/results1.txt";

# randomly permutate @array in place
sub fisher_yates_shuffle
{
    my $array = shift;
    my $i = @$array;
    while ( --$i )
    {
        my $j = int rand( $i+1 );
        @$array[$i,$j] = @$array[$j,$i];
    }
}

# shuffle
fisher_yates_shuffle( \@record_nums );  

# write to file
open my $fp, '>', $record_fname;
print $fp "$nrecords\n";
print $fp "$ntrain\n";
print $fp "$ntest\n";
print $fp "$cookie\n";
foreach (@record_nums) { 
	if($_ != $trecord){
    	print $fp "$_\n";
	}
} 
close($fp);



print "running clean up..\n";
$cmd = "rm data/output_*;";
system($cmd);

print "run MRA classification..\n";
foreach (@data_files){ 

	print("batch process..\n");
	$cmd = "./batch_process.pl $_ $test_labels_fname $train_labels_fname $record_fname";
    print("$cmd\n");
	system($cmd);


	$fname_train = "data/train_$_";
	$fname_train_csv = $fname_train;
	$fname_train_csv =~ s/txt/csv/g;
	$fname_train_arff = $fname_train;
	$fname_train_arff =~ s/txt/arff/g;
	
	$fname_test = "data/test_$_";
	$fname_test_csv = $fname_test;
	$fname_test_csv =~ s/txt/csv/g;
	$fname_test_arff = $fname_test;
	$fname_test_arff =~ s/txt/arff/g;
	$fname_out = "data/output_aw_$_";

    print("run Java Auto-Weka classifier (with time limit)..\n");
    $cmd = "java -cp \":weka-3-8-1/weka.jar\" -cp \":autoweka_files/autoweka.jar\" runAW2 $fname_train_arff $fname_test_arff $fname_out"; 
	print "$cmd\n";
	system($cmd);
}

#print "run LSTM classifier\n";
#$cmd = "python3 lstm1_sound.py";
#system($cmd);

print "==> make combined predictions..\n";
my $data_files_str = join(' ',@data_files);
$cmd_args = $predicted_labels_fname." ".scalar(@data_files)." ".$data_files_str;
$cmd = "./make_combined_predictions1.py $cmd_args";
print("$cmd\n");
system($cmd);
print("==> end make_combined_predictions.py\n");

print "print results..\n";
print "test/train split used: $ntest / $ntrain\n";
$cmd_args = $test_labels_fname." ".$predicted_labels_fname;
$cmd = "./analyze_results.py $cmd_args";
print("$cmd\n");
system($cmd);

