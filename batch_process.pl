#!/usr/bin/perl

my $samples_fname_txt = $ARGV[0];
my $test_labels_fname_txt = $ARGV[1];
my $train_labels_fname_txt = $ARGV[2];
my $record_nums_fname_txt = $ARGV[3];

my $samples_fname_csv = $samples_fname_txt;
$samples_fname_csv =~ s/txt/csv/g;
my $ddir = "data";

my $test_fname_csv = "$ddir/test_".$samples_fname_csv; 
my $train_fname_csv = "$ddir/train_".$samples_fname_csv; 

$samples_fname_txt = "$ddir/$samples_fname_txt";
$samples_fname_csv = "$ddir/$samples_fname_csv";

# read records
open(FILE, "<", $samples_fname_txt) or die("Can't open $samples_fname_txt");
@records = <FILE>;
close(FILE);
chomp(@records);
# remove header
$header = $records[0];
splice(@records, 0, 1);
print "length records array: ", scalar(@records), "\n";


# read record nums for test/train split info
open(FILE, "<", $record_nums_fname_txt) or die("Can't open $record_nums_fname_txt");
@record_nums = <FILE>;
close(FILE);
chomp(@record_nums);
$nrecords = $record_nums[0];
$ntrain = $record_nums[1];
$ntest = $record_nums[2];
#splice(@record_nums, 0, 1);
#splice(@record_nums, 1, 1);
#splice(@record_nums, 2, 1);
print "nrecords = $nrecords; ntrain = $ntrain; ntest = $ntest\n";
print "length record nums array: ", scalar(@records), "\n";


my @train;
my @test;
print "splitting: $ntrain, $ntest\n";
for($i=0; $i<$ntrain; $i++){
    $lnum = $record_nums[$i];
    if(length($records[$lnum])>1){
        push @train, $records[$lnum]; 
    }
}

for($i=$ntrain; $i<$nrecords; $i++){
    $lnum = $record_nums[$i];
    if(length($records[$lnum])>1){
        push @test, $records[$lnum]; 
    }
}


print "length test array: ", scalar(@test), "\n";
print "length train array: ", scalar(@train), "\n";




print "length train array: ", scalar(@train), "\n";
#print join(", ", @train);
#for my $line (@train) { print " ",length($line); }
print "\n";
open(FILE, ">", $train_fname_csv) or die("Can't open file");
print FILE "$header\n";
#for my $line (@train) { print FILE "$line\n"; }
for my $line (@train) { if(length($line)>1){print FILE "$line\n";} }
close(FILE);

# check to make sure we have the train records we expect
my $train_cnt = 0;
open(FILE,$train_fname_csv) or die "can't open $!";
$train_cnt++ while <FILE>;
close FILE;
$train_cnt--; # header
print "train_cnt = $train_cnt\n";

print "length test array: ", scalar(@test), "\n";
#for my $line (@test) { print " ",length($line); }
print "\n";
open(FILE, ">", $test_fname_csv) or die("Can't open file");
print FILE "$header\n";
for my $line (@test) { if(length($line)>1){print FILE "$line\n";} }
close(FILE);

# check to make sure we have the test records we expect
my $test_cnt = 0;
open(FILE,$test_fname_csv) or die "can't open $!";
$test_cnt++ while <FILE>;
close FILE;
$test_cnt--; # header
print "test_cnt = $test_cnt\n";

# extract and save the test labels and train labels
# make sure we extract the number correspoding to test csv
open(FILE, ">", $test_labels_fname_txt) or die("Can't open file");
for($i=0; $i<$test_cnt; $i++) {
    $label = substr($test[$i], 0, index($test[$i], ','));
    $label =~ s/class_//; 
    if(length($label)==0){$label=1}
    print FILE "$label\n";
}
close(FILE);

open(FILE, ">", $train_labels_fname_txt) or die("Can't open file");
for($i=0; $i<$train_cnt; $i++) {
    $label = substr($train[$i], 0, index($train[$i], ','));
    $label =~ s/class_//; 
    if(length($label)==0){$label=1}
    print FILE "$label\n";
}
close(FILE);


# convert to arff
$train_fname_arff = $train_fname_csv;
$train_fname_arff =~ s/csv/arff/g;

$test_fname_arff = $test_fname_csv;
$test_fname_arff =~ s/csv/arff/g;

$cmd = "java -cp \":../weka-3-8-1/weka.jar\" CSV2Arff $train_fname_csv $train_fname_arff";
print "$cmd\n";
system($cmd);

$cmd = "java -cp \":../weka-3-8-1/weka.jar\" CSV2Arff $test_fname_csv $test_fname_arff";
print "$cmd\n";
system($cmd);


open my $fp, '<', $train_fname_arff;
chomp(my @records = <$fp>);
close $fp;

open my $fp, '>', $train_fname_arff;
foreach (@records) { 
    if($_=~m/attribute label/i){
        print $fp "\@attribute label {class_1,class_2,class_3,class_4,class_5,class_6,class_7,class_8,class_9,class_10}\n";
    }
    else{
        print $fp "$_\n";
    }
} 

open my $fp, '<', $test_fname_arff;
chomp(my @records = <$fp>);
close $fp;

open my $fp, '>', $test_fname_arff;
foreach (@records) { 
    if($_=~m/attribute label/i){
        print $fp "\@attribute label {class_1,class_2,class_3,class_4,class_5,class_6,class_7,class_8,class_9,class_10}\n";
    }
    else{
		# remove true label from test set
		$_ =~ s/class_2/class_1/g;
		$_ =~ s/class_3/class_1/g;
		$_ =~ s/class_4/class_1/g;
		$_ =~ s/class_5/class_1/g;
		$_ =~ s/class_6/class_1/g;
		$_ =~ s/class_7/class_1/g;
		$_ =~ s/class_8/class_1/g;
		$_ =~ s/class_9/class_1/g;
		$_ =~ s/class_10/class_1/g;
        print $fp "$_\n";
    }
} 

