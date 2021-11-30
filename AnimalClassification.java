package ai.certifai.AnimalClassification;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.KFoldIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class AnimalClassification {

    static final double ratio = 0.8;
    static final int seed = 1234;
    static  final double lr = 0.2;
    static final int numOutputs= 7;
    static final int numInputs= 102;
    static final int epoch= 10;
    static INDArray weightsArray = Nd4j.create(new double[]{0.4, 0.5, 1.0, 0.5, 0.5, 0.4,0.4});

    public static void main(String[] args) throws  Exception {

        //loadfile
        File myfile =  new ClassPathResource("zoo.csv").getFile();

        //FileSplit
        FileSplit fileSplit = new FileSplit(myfile);

        //set CSV Record Reader and initialize it
        RecordReader rr = new CSVRecordReader(1,',');
        rr.initialize(fileSplit);

        //Build Schema
        Schema schema = new Schema.Builder()
                .addColumnsString("Animal_names")
                .addColumnCategorical("hair", Arrays.asList("0","1"))
                .addColumnCategorical("feathers", Arrays.asList("0", "1"))
                .addColumnCategorical("eggs", Arrays.asList("0", "1"))
                .addColumnCategorical("milk", Arrays.asList("0", "1"))
                .addColumnCategorical("airbone", Arrays.asList("0","1"))
                .addColumnCategorical("aquatic", Arrays.asList("0","1"))
                .addColumnCategorical("predator", Arrays.asList("0","1"))
                .addColumnCategorical("toothed", Arrays.asList("0","1"))
                .addColumnCategorical("backbone", Arrays.asList("0","1"))
                .addColumnCategorical("breathes", Arrays.asList("0","1"))
                .addColumnCategorical("venomous", Arrays.asList("0","1"))
                .addColumnCategorical("tail", Arrays.asList("0","1"))
                .addColumnCategorical("domestic", Arrays.asList("0","1"))
                .addColumnCategorical("catsize", Arrays.asList("0","1"))
                .addColumnCategorical("class_type", Arrays.asList("0","1","2","3","4","5","6","7"))
                .addColumnInteger("legs")
                .build();

        //Build TransformProcess to transform the data
        TransformProcess transform = new TransformProcess.Builder(schema)
                .build();

        // Checking the schema
        Schema outputSchema = transform.getFinalSchema();
        System.out.println(outputSchema);

        List<List<Writable>> allData = new ArrayList<>();

        while(rr.hasNext()){
            allData.add(rr.next());
        }

        List<List<Writable>> processData = LocalTransformExecutor.execute(allData, transform);

        //Create iterator from process data
        CollectionRecordReader collectionRR = new CollectionRecordReader(processData);

        //Input batch size , label index , and number of label
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(collectionRR, processData.size(),-1,7);

        //Create Iterator and shuffle the dat
        DataSet fullDataset = dataSetIterator.next();
        fullDataset.shuffle(seed);

//      //Input split ratio
        SplitTestAndTrain testAndTrain = fullDataset.splitTestAndTrain(ratio);
//
        //Get train and test dataset
        DataSet trainData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        //printout size
        System.out.println("Training vector : ");
        System.out.println(Arrays.toString(trainData.getFeatures().shape()));
        System.out.println("Test vector : ");
        System.out.println(Arrays.toString(testData.getFeatures().shape()));

        //Data normalization
        DataNormalization normalizer = new NormalizerMinMaxScaler();
        normalizer.fit(trainData);
        normalizer.transform(trainData);
        normalizer.transform(testData);

        //Network config

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(lr, Nesterovs.DEFAULT_NESTEROV_MOMENTUM))
                .l2(0.001)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(100)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(100)
                        .nOut(75)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nIn(75)
                        .nOut(90)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new DenseLayer.Builder()
                        .nIn(90)
                        .nOut(150)
                        .activation(Activation.RELU)
                        .build())
                .layer(4, new OutputLayer.Builder()
                        .nIn(40)
                        .nOut(numOutputs)
                        .lossFunction(new LossMCXENT(weightsArray))
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();



        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        //UI-Evaluator
        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);

        //Set model listeners
        model.setListeners(new StatsListener(storage, 10));


        //Training
        Evaluation eval;
        for(int i=0; i < epoch; i++) {
            model.fit(trainData);
            eval = model.evaluate(new ViewIterator(testData, processData.size()));
            System.out.println("EPOCH: " + i + " Accuracy: " + eval.accuracy());
        }


        //Confusion matrix
        Evaluation evalTrain = model.evaluate(new ViewIterator(trainData, processData.size()));
        Evaluation evalTest = model.evaluate(new ViewIterator(testData,processData.size()));
        System.out.print("Train Data");
        System.out.println(evalTrain.stats());

        System.out.print("Test Data");
        System.out.print(evalTest.stats());
    }
}
