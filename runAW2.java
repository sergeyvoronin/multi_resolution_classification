import weka.classifiers.AbstractClassifier;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.core.converters.ConverterUtils; 
import weka.core.Instances;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest; 
import weka.classifiers.Classifier; 
import java.io.File;
import java.io.PrintWriter;
import weka.core.Attribute;
import java.util.ArrayList;
import java.util.Random;
import java.io.Serializable;
import java.io.ObjectInputStream;
import java.io.FileInputStream; 

// run auto-weka on test train sets..

//import weka.core.*;
import weka.classifiers.evaluation.output.prediction.PlainText;

//import autoweka
import weka.classifiers.meta.AutoWEKAClassifier; 


public class runAW2 {
    public static void main(String[] args) throws Exception {
        int i,c;

        ConverterUtils.DataSource source1 = new ConverterUtils.DataSource(args[0]);
        Instances train = source1.getDataSet();
        if (train.classIndex() == -1)
            train.setClassIndex(0);

        ConverterUtils.DataSource source2 = new ConverterUtils.DataSource(args[1]);
        Instances test = source2.getDataSet();
        if (test.classIndex() == -1)
            test.setClassIndex(0);

		String output_file = args[2];
		System.out.printf("train set: %s\n", args[0]);
		System.out.printf("test set: %s\n", args[1]);
        System.out.printf("num instances train = %d\n", train.numInstances());
        System.out.printf("num instances test = %d\n", test.numInstances());
        System.out.printf("output_file = %s\n", output_file);

    
        // init object and set options
        AutoWEKAClassifier awc = new AutoWEKAClassifier();
        // set time limit
        awc.setTimeLimit(15);
        // optional, set mem limit

        // train classifier
        awc.buildClassifier(train);

        // print predictions 
        PrintWriter writer = new PrintWriter(output_file, "UTF-8");
        writer.println("prediction,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10");
        for(i=0; i<test.numInstances(); i++){
            double label = awc.classifyInstance(test.instance(i));
            double[] prob = awc.distributionForInstance(test.instance(i));
            test.instance(i).setClassValue(label);
            writer.printf("%s,", test.instance(i).stringValue(0));
            writer.printf("%f,", prob[0]);
            writer.printf("%f\n", prob[1]);
            writer.printf("%f\n", prob[2]);
            writer.printf("%f\n", prob[3]);
            writer.printf("%f\n", prob[4]);
            writer.printf("%f\n", prob[5]);
            writer.printf("%f\n", prob[6]);
            writer.printf("%f\n", prob[7]);
            writer.printf("%f\n", prob[8]);
            writer.printf("%f\n", prob[9]);
        }
        writer.close();
	}
}

