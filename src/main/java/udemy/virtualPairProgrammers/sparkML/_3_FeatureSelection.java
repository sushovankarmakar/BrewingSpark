package udemy.virtualPairProgrammers.sparkML;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class _3_FeatureSelection {

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

        /*csvDataset.describe()
                .show(false);*/

        // dropping those columns which are irrelevant
        csvDataset = csvDataset.drop("id", "date", "waterfront", "view", "condition", "grad", "yr_renovated", "zipcode", "lat", "long");

        for (String col : csvDataset.columns()) {
            double correlation = csvDataset.stat().corr("price", col);
            System.out.println("The correlation between the price and " + col + " is " + correlation);
        }

        // dropping those columns which have low correlation with 'price' column
        csvDataset = csvDataset.drop("sqft_lot", "yr_built", "sqft_lot15", "sqft_living15");

        // building a feature correlation matrix
        for (String col1 : csvDataset.columns()) {
            for (String col2 : csvDataset.columns()) {
                double correlation = csvDataset.stat().corr(col1, col2);
                System.out.println("The correlation between the " + col1 + " and " + col2 + " is " + correlation);
            }
        }
    }
}
