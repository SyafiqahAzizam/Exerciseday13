package global.skymind.question1;

/*
 Collected a dataset on how different shape of buildings affect the energy efficiency.

 * Dataset origin: https://archive.ics.uci.edu/ml/datasets/Energy+efficiency#
 * Dataset attribute description:
 * X1: Relative Compactness
 * X2: Surface Area
 * X3: Wall Area
 * X4: Roof Area
 * X5: Overall Height
 * X6: Orientation
 * X7: Glazing Area
 * X8: Glazing Area Distribution
 * Y1: Heating Load
 * Y2: Cooling Load
 */

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
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class EnergyEfficiency {

    //===============Tunable parameters============
    private static int batchSize = 50;
    private static int seed = 123;
    private static double trainFraction = 0.8;
    private static double lr = 0.001;
    //=============================================
    private static TransformProcess tp;
    private static DataSet trainSet;
    private static DataSet testSet;
    private static RegressionEvaluation evalTrain;
    private static RegressionEvaluation evalTest;

    private static int skipline = 1;
    private static int numOfEpochs;



    public static void main(String[] args) throws IOException, InterruptedException {

        File filePath = new ClassPathResource("EnergyEfficiency/ENB2012_data.csv").getFile();

        //Split
        FileSplit filesplit = new FileSplit(filePath);

        //RecorReader
        RecordReader recordReader = new CSVRecordReader(skipline, ',');
        recordReader.initialize(filesplit);


        Schema schema = new Schema.Builder()
                .addColumnsDouble("rel-compactness","surface-area","wall-area","roof-area","overall-height")
                .addColumnInteger("orientation")
                .addColumnDouble("glazing-area")
                .addColumnInteger("glazing-area-distribution")
                .addColumnsDouble("heating-load","cooling-load")
                .addColumnsString("emptyCol1","emptyCol2")
                .build();

        //Transform Process
        tp = new TransformProcess.Builder(schema)
                .removeColumns("emptyCol1", "emptyCol2")
                .filter(new FilterInvalidValues())//filter row kosong tapi tatau row mana kosong
                .build();

        List<List<Writable>> data = new ArrayList<>();

        while(recordReader.hasNext())
        {
            data.add(recordReader.next());
        }

        List<List<Writable>> transformed = LocalTransformExecutor.execute(data, tp);
        System.out.println("=======Initial Schema=========\n"+ tp.getInitialSchema());
        System.out.println("=======Final Schema=========\n"+ tp.getFinalSchema());


        System.out.println("Size before transform: "+ data.size()+
                "\nColumns before transform: "+ tp.getInitialSchema().numColumns());
        System.out.println("Size after transform: "+ transformed.size()+
                "\nColumns after transform: "+ tp.getFinalSchema().numColumns());

        CollectionRecordReader crr = new CollectionRecordReader(transformed);
        DataSetIterator iter = new RecordReaderDataSetIterator(crr, transformed.size(), 8, 9, true ) //why not batch, bcs after this we want shuffle the whole data not a certain batch
                                                                                                    //column 8,9 for output

        DataSet dataSet = iter.next();
        dataSet.shuffle(seed);

        SplitTestAndTrain testAndTrain = dataSet.splitTestAndTrain(trainFraction);

        trainSet= testAndTrain.getTrain();
        testSet = testAndTrain.getTest();

        DataNormalization normalizer = new NormalizerMinMaxScaler(); // NormaliserStandardise()
        scaler.fitLabel(true); //for regression, skill the target as well rather than features
        normalizer.fit(trainSet);
        normalizer.transform(trainSet);
        normalizer.transform(testSet);

        ViewIterator trainIter = new ViewIterator(trainSet, batchSize);
        ViewIterator testIter = new ViewIterator(testSet, batchSize);

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .updater(new Adam(lr))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(trainIter.inputColumns())
                        .nOut(50)
                        .nOut(50)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nOut(2)
                        .activation(Activation.IDENTITY)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        EarlyStoppingConfiguration esConfig = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(new ScoreImprovementEpochTermination(1)) //terminate if no improvement
                .scoreCalculator(new DataSetLossCalculator(testIter,true))
                .evaluateEveryNEpochs(1)
                .build();

        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConfig,config,trainIter );
        EarlyStoppingResult result = trainer.fit();


        System.out.println("Score at best epoch: " + result.getBestModelScore());

        //UI
        StatsStorage statsStorage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(statsStorage);

        //model listeners
        model.setListeners(new ScoreIterationListener(10), new StatsListener(statsStorage, 10));

        // for model evaluation during training
        Evaluation evalTrainSet;
        Evaluation evalValidSet;
        DataSetLossCalculator trainLossCalculator = new DataSetLossCalculator(trainIter, true);
        DataSetLossCalculator validLossCalculator = new DataSetLossCalculator(testIter, true);
        ArrayList<Double> trainLoss = new ArrayList<>();
        ArrayList<Double> testLoss = new ArrayList<>();
        numOfEpochs = result.getBestModelEpoch(); // set the optimal number of epochs



        // fit model
        for(int i = 0; i<numOfEpochs; i++){
            model.fit(trainIter);

            // calculate training loss
            trainLoss.add(trainLossCalculator.calculateScore(model));
            // calculate test loss
            testLoss.add(validLossCalculator.calculateScore(model));

            evalTrainSet = model.evaluate(trainIter);
            evalValidSet = model.evaluate(testIter);
            System.out.println("EPOCH: " + i + ", Train f1: " + evalTrainSet.f1());
            System.out.println("EPOCH: " + i + ", Validation f1: " + evalValidSet.f1());

            trainIter.reset();
            testIter.reset();
        }

        evalTrain= model.evaluateRegression(testIter);
        System.out.println(evalTrain.stats());
        evalTest= model.evaluateRegression(testIter);
        System.out.println(evalTest.stats());

    }

}
