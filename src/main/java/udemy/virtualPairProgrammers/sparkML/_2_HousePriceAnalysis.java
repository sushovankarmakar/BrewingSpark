package udemy.virtualPairProgrammers.sparkML;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.param.DoubleParam;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionSummary;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class _2_HousePriceAnalysis {

    public static void main(String[] args) {

        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkConf sparkConf = new SparkConf()
                .setAppName("house-price-analysis")
                .setMaster("local[*]");

        SparkSession sparkSession = SparkSession.builder()
                .config(sparkConf)
                .getOrCreate();

        Dataset<Row> csvDataset = sparkSession.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("src/main/resources/udemy/virtualPairProgrammers/sparkML/kc_house_data.csv");

        //csvDataset.printSchema();

        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(new String[]{"bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "grade"})
                .setOutputCol("features");

        Dataset<Row> modelInputData = vectorAssembler.transform(csvDataset)
                .select("price", "features")
                .withColumnRenamed("price", "label");

        //modelInputData.show(false);

        Dataset<Row>[] inputData = modelInputData.randomSplit(new double[]{0.8, 0.2});
        Dataset<Row> trainingAndTestData = inputData[0];
        Dataset<Row> holdOutData = inputData[1];

        // 1. running process one time with model fitting parameters
        /*LinearRegressionModel linearRegressionModel = new LinearRegression()
                .setMaxIter(10) // these are model fitting parameters
                .setRegParam(0.3)
                .setElasticNetParam(0.8)
                .fit(trainingData);*/

        // 2. running process multiple time with model fitting parameters
        LinearRegression linearRegression = new LinearRegression();

        // Using this ParamMap, we're asking Spark to run the model building process lots of time
        ParamGridBuilder paramGridBuilder = new ParamGridBuilder();
        ParamMap[] paramMap = paramGridBuilder
                .addGrid(linearRegression.regParam(), new double[]{0.01, 0.1, 0.5})
                .addGrid(linearRegression.elasticNetParam(), new double[]{0, 0.5, 1})
                .build();

        TrainValidationSplit trainValidationSplit = new TrainValidationSplit()
                .setEstimator(linearRegression)
                .setEvaluator(new RegressionEvaluator().setMetricName("r2"))
                .setEstimatorParamMaps(paramMap)
                .setTrainRatio(0.8);

        TrainValidationSplitModel trainValidationSplitModel = trainValidationSplit.fit(trainingAndTestData); // TrainValidationSplitModel is wrapping the LinearRegressionModel
        LinearRegressionModel linearRegressionModel = (LinearRegressionModel) trainValidationSplitModel.bestModel();

        double intercept = linearRegressionModel.intercept();
        Vector coefficients = linearRegressionModel.coefficients();

        double regParam = linearRegressionModel.getRegParam();
        double elasticNetParam = linearRegressionModel.getElasticNetParam();

        System.out.println("The model has intercept: " + intercept + ", \n" +
                "coefficients: " + coefficients + ", \n" +
                "regParam: " + regParam + ", \n" +
                "elasticNetParam: " + elasticNetParam);

        double r2 = linearRegressionModel.summary().r2();
        double rootMeanSquaredError = linearRegressionModel.summary().rootMeanSquaredError();
        System.out.println("The training data has r2 " + r2 + " and rootMeanSquaredError " + rootMeanSquaredError);

        linearRegressionModel.transform(holdOutData).show(false);

        LinearRegressionSummary evaluatedData = linearRegressionModel.evaluate(holdOutData);
        System.out.println("The holdout data has r2 " + evaluatedData.r2() + " and rootMeanSquaredError " + evaluatedData.rootMeanSquaredError());
    }
}
