#!/usr/bin/python3
import pandas as pd
import numpy as np
import glob
import os, sys
import scipy.stats as stat

def main(argv):
        output_fname = sys.argv[1]
        num_files = int(sys.argv[2]);
        main_fname = "data/output_aw_" + sys.argv[3];
        out_main = pd.read_csv(main_fname)
        out_main_vals = out_main.values;
        num_vals = len(out_main);

        #out_nn = np.loadtxt('data/output_nn.txt');
        #out_nn  = [int(x) for x in out_nn]
        
        fp = open(output_fname, 'w')

        for i in range(0,num_vals):
            print('---> processing entry %d' % i);
            rvals = out_main_vals[i];
            max_prob = np.max(rvals[1:10]);
            master_guess = int(rvals[0].replace('class_',''));
            my_list = []
            print('master_guess = %d' % master_guess);
            my_list.append(master_guess)
            my_list.append(master_guess)
            for l in range(4,num_files+2):
                fname = "data/output_aw_" + sys.argv[l];
                out_other = pd.read_csv(fname)
                out_other_vals = out_other.values;
                rvals_other = out_other_vals[i];
                max_prob_other = np.max(rvals_other[1:10]);
                class_guess = int(rvals_other[0].replace('class_',''));
                print('class_guess = %d' % class_guess);
                #if max_prob_other > 0.65:
                my_list.append(class_guess)
            
            if len(my_list) > 2:
                output_guess = stat.mode(my_list)
                output_guess = output_guess[0];
                output_guess = output_guess[0];
            else:
                output_guess = master_guess;
            print(my_list);
            print('pick mode: class %d' % output_guess);
        
            fp.write("%d\n" % output_guess);
        
        # close and exit    
        fp.close();


if __name__ == "__main__":
	main(sys.argv[1:])
