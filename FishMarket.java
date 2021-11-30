package ai.certifai.weightFishMarket;


import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.common.io.ClassPathResource;
import org.datavec.api.transform.schema.Schema;
import org.nd4j.evaluation.classification.Evaluation;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.KFoldIterator;

import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;


import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


// a record of 7 common different fish species in fish market sales.
// a predictive model can be performed using machine friendly data
// estimated weight of fishes can be predicted.

public class FishMarket {

    static int seed = 1234;
    static double lr = 0.01;
    static int numInp =7;
    static int numOut =7;
    static int epoch = 500;
    public static final String BLACK_BOLD = "\033[1;30m";
    public static final String BLUE_BOLD = "\033[1;34m";
    public static final String ANSI_RESET = "\u001B[0m";


    public static void main(String[] args) throws Exception{

        //set filepath
        File myfile = new ClassPathResource("Fish.csv").getFile();

        //Split
        FileSplit fileSplit = new FileSplit(myfile);

        //set CSV Record Reader and initialize it
        RecordReader rr = new CSVRecordReader(1,',');
        rr.initialize(fileSplit);

        //Build Schema
        Schema schema = new Schema.Builder()
                .addColumnCategorical("Species", Arrays.asList("Bream","Parkki","Porch","Roach","Whitefish","Smelt","Perch"))
                .addColumnsFloat("Weight","Length1","Length2","Height","Width")
                .build();

        //Build TransformProcess to transform the data
        TransformProcess transform = new TransformProcess.Builder(schema)
                .categoricalToInteger("Species")
                .build();

        List<List<Writable>> data = new ArrayList<>();
        while(rr.hasNext())
        {
            data.add(rr.next());
        }
        List<List<Writable>> processData = LocalTransformExecutor.execute(data, transform);

        //Create iterator from process data
        CollectionRecordReader collectionRR = new CollectionRecordReader(processData);

        //Input batch size , label index , and number of label
        DataSetIterator iter = new RecordReaderDataSetIterator(collectionRR, processData.size(),1,7, true);

        //Create Iterator and shuffle the dat
        DataSet fullDataset = iter.next();
        fullDataset.shuffle(seed);

        //create kfold iterator (k=5)
        KFoldIterator kFoldIterator = new KFoldIterator(5, fullDataset);


        //Network config

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(lr))
                .l2(0.2)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(numInp)
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(128)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder()
                        .nIn(64)
                        .nOut(numOut)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .build())
                .build();

        //start the kfold evaluation
        int i = 1;


        //initialize an empty list to store the F1 score
        ArrayList<Double> f1List = new ArrayList<>();

        //for each fold
        while (kFoldIterator.hasNext()) {
            System.out.println(BLACK_BOLD + "Fold: " + i + ANSI_RESET);

            //for each fold, get the features and labels from training set and test set
            DataSet currDataSet = kFoldIterator.next();
            INDArray trainFoldFeatures = currDataSet.getFeatures();
            INDArray trainFoldLabels = currDataSet.getLabels();
            INDArray testFoldFeatures = kFoldIterator.testFold().getFeatures();
            INDArray testFoldLabels = kFoldIterator.testFold().getLabels();
            DataSet trainDataSet = new DataSet(trainFoldFeatures, trainFoldLabels);

            //scale the dataset
            NormalizerMinMaxScaler scaler = new NormalizerMinMaxScaler();
            scaler.fit(trainDataSet);
            scaler.transform(trainFoldFeatures);
            scaler.transform(testFoldFeatures);

            //initialize the model
            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();

            //train the data
            for (int j = 0; j < epoch; j++) {
                model.fit(trainDataSet);
            }

            //evaluate the model with test set
            Evaluation eval = new Evaluation();
            eval.eval(testFoldLabels, model.output(testFoldFeatures));

            //print out the evaluation results
            System.out.println(eval.stats());
            //save the eval results
            f1List.add(eval.f1());

            i++;

        }

        INDArray f1scores = Nd4j.create(f1List);
        System.out.println("Average F1 scores for all folds: " + f1scores.mean(0));



    }}
