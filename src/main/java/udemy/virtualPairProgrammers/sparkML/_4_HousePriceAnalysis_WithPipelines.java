package udemy.virtualPairProgrammers.sparkML;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.ml.*;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
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
import org.apache.spark.sql.functions;

public class _4_HousePriceAnalysis_WithPipelines {

    public static void main(String[] args) {

        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkConf sparkConf = new SparkConf()
                .setAppName("house-price-analysis-with-pipelines")
                .setMaster("local[*]");

        SparkSession sparkSession = SparkSession.builder()
                .config(sparkConf)
                .getOrCreate();

        Dataset<Row> csvDataset = sparkSession.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("src/main/resources/udemy/virtualPairProgrammers/sparkML/kc_house_data.csv");

        //csvDataset.printSchema();

        csvDataset = csvDataset.withColumn("sqrt_above_percentage",
                        functions.col("sqft_above").divide(functions.col("sqft_living")))
                .withColumnRenamed("price", "label");

        Dataset<Row>[] inputData = csvDataset.randomSplit(new double[]{0.8, 0.2});
        Dataset<Row> trainingAndTestData = inputData[0];
        Dataset<Row> holdOutData = inputData[1];

        // condition -> conditionIndex -> conditionVector
        // grade -> gradeIndex -> gradeVector
        // zipcode -> zipcodeIndex -> zipcodeVector
        StringIndexer conditionIndexer = new StringIndexer()
                .setInputCol("condition")
                .setOutputCol("conditionIndex");

        StringIndexer gradeIndexer = new StringIndexer()
                .setInputCol("grade")
                .setOutputCol("gradeIndex");

        StringIndexer zipCodeIndexer = new StringIndexer()
                .setInputCol("zipcode")
                .setOutputCol("zipcodeIndex");

        OneHotEncoder encoder = new OneHotEncoder()
                .setInputCols(new String[]{"conditionIndex", "gradeIndex", "zipcodeIndex"})
                .setOutputCols(new String[]{"conditionVector", "gradeVector", "zipcodeVector"});

        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(new String[]{"bedrooms", "bathrooms", "sqft_living",
                        "sqrt_above_percentage", "floors",
                        "conditionVector", "gradeVector", "zipcodeVector"})
                .setOutputCol("features");

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

        // ----------------------------- AREA TO FOCUS ------------------------------------------
        //  In production env set up, Pipeline helps us to chain all of those methods calls into a single method call.

        // conditionIndexer -> gradeIndexer -> zipCodeIndexer -> encoder -> vectorAssembler -> trainValidationSplit
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{
                        conditionIndexer, gradeIndexer, zipCodeIndexer, encoder, vectorAssembler, trainValidationSplit
                }); // order is important, execution order will be the way we've mentioned

        PipelineModel pipelineModel = pipeline.fit(trainingAndTestData);
        TrainValidationSplitModel model = (TrainValidationSplitModel) pipelineModel.stages()[5];
        LinearRegressionModel lrModel = (LinearRegressionModel) model.bestModel();

        // -----------------------------------------------------------------------

        Dataset<Row> holdOutResults = pipelineModel.transform(holdOutData)
                .drop("prediction"); // dropping this column to mitigate this exception: 'requirement failed: Column prediction already exists.'
        holdOutResults.show(false);

        double intercept = lrModel.intercept();
        Vector coefficients = lrModel.coefficients();

        double regParam = lrModel.getRegParam();
        double elasticNetParam = lrModel.getElasticNetParam();

        System.out.println("The model has intercept: " + intercept + ", \n" +
                "coefficients: " + coefficients + ", \n" +
                "regParam: " + regParam + ", \n" +
                "elasticNetParam: " + elasticNetParam);

        double r2 = lrModel.summary().r2();
        double rootMeanSquaredError = lrModel.summary().rootMeanSquaredError();
        System.out.println("The training data has r2 " + r2 + " and rootMeanSquaredError " + rootMeanSquaredError);

        LinearRegressionSummary evaluatedData = lrModel.evaluate(holdOutResults);
        System.out.println("The holdout data has r2 " + evaluatedData.r2() + " and rootMeanSquaredError " + evaluatedData.rootMeanSquaredError());

    }
}
