package udemy.virtualPairProgrammers.sparkML;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionSummary;
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

        Dataset<Row>[] trainingAndTestData = modelInputData.randomSplit(new double[]{0.8, 0.2});
        Dataset<Row> trainingData = trainingAndTestData[0];
        Dataset<Row> testData = trainingAndTestData[1];

        LinearRegressionModel linearRegressionModel = new LinearRegression().fit(trainingData);

        double intercept = linearRegressionModel.intercept();
        Vector coefficients = linearRegressionModel.coefficients();

        System.out.println("The model has intercept " + intercept + " and coefficients " + coefficients);

        double r2 = linearRegressionModel.summary().r2();
        double rootMeanSquaredError = linearRegressionModel.summary().rootMeanSquaredError();
        System.out.println("The training data has r2 " + r2 + " and rootMeanSquaredError " + rootMeanSquaredError);

        //linearRegressionModel.transform(testData).show(false);

        LinearRegressionSummary evaluatedData = linearRegressionModel.evaluate(testData);
        System.out.println("The test data has r2 " + evaluatedData.r2() + " and rootMeanSquaredError " + evaluatedData.rootMeanSquaredError());
    }
}
