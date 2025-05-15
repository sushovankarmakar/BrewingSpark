import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import javax.crypto.Cipher;
import javax.crypto.spec.SecretKeySpec;
import java.util.Base64;

public class CsvToOrcWithEncryption {

    private static final String KEY = "1234567890123456"; // 16-byte key for AES-128
    private static final SecretKeySpec KEY_SPEC = new SecretKeySpec(KEY.getBytes(), "AES");

    public static void main(String[] args) {

        // Initialize SparkSession
        SparkSession spark = SparkSession.builder()
                .appName("CsvToOrcWithEncryption")
                .config("spark.master", "local")
                .getOrCreate();

        // HDFS paths
        String inputPath = "/data/poc_input/biodata.csv";
        String outputPath = "/data/poc_output/";

        StructType schema = new StructType(new StructField[]{
                DataTypes.createStructField("name", DataTypes.StringType, true),
                DataTypes.createStructField("age", DataTypes.IntegerType, true),
                DataTypes.createStructField("phone_number", DataTypes.StringType, true),
        });

        // Read CSV file
        Dataset<Row> df = spark.read()
                .format("csv")
                .option("header", "true")
                .schema(schema)
                .load(inputPath);

        df.show(false);

        // Encrypt phone_number column
        /*Dataset<Row> encryptedDf = df.withColumn("phone_number", functions.udf(
                (String phoneNumber) -> encrypt(phoneNumber), DataTypes.StringType).apply(df.col("phone_number")));*/

        // Write to ORC format
        df.write()
                .format("orc")
                .option("compression", "zstd")
                .option("orc.stripe.size", 10)
                .mode(SaveMode.Overwrite)
                .save(outputPath);

        spark.stop();
    }

    // Encryption method
    private static String encrypt(String value) {

        try {
            // Cipher setup
            Cipher cipher = Cipher.getInstance("AES");
            cipher.init(Cipher.ENCRYPT_MODE, KEY_SPEC);

            // Encryption
            byte[] encrypted = cipher.doFinal(value.getBytes());
            return Base64.getEncoder().encodeToString(encrypted);

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
