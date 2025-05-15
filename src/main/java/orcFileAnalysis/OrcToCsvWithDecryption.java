package orcFileAnalysis;

import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.crypto.Cipher;
import javax.crypto.spec.SecretKeySpec;
import java.util.Base64;

public class OrcToCsvWithDecryption {

    private static final String KEY = "1234567890123456";
    //private static final String KEY = "1234567890123456-1"; // WRONG KEY
    private static final SecretKeySpec KEY_SPEC = new SecretKeySpec(KEY.getBytes(), "AES");

    private static final Logger log = LoggerFactory.getLogger(OrcToCsvWithDecryption.class);

    public static void main(String[] args) {
        // Initialize SparkSession
        SparkSession spark = SparkSession.builder()
                .appName("orcFileAnalysis.OrcToCsvWithDecryption")
                .config("spark.master", "local")
                .getOrCreate();

        // HDFS paths
        String inputPath = "/data/poc_output/biodata_encrypted.orc";
        String outputPath = "/data/poc_output_decrypted/biodata_decrypted.csv";

        // Read ORC file
        Dataset<Row> df = spark.read()
                .format("orc")
                .load(inputPath);

        System.out.println("--------- Data in encrypted format ---------");
        df.show(false);

        // Register UDF for decryption
        spark.udf().register("decrypt", (String encryptedPhoneNumber) -> decrypt(encryptedPhoneNumber), DataTypes.StringType);

        // Apply decryption UDF to phone_number column
        Dataset<Row> decryptedDf = df.withColumn("phone_number", functions.callUDF("decrypt", df.col("phone_number")));

        // Show contents in the console
        try {
            System.out.println("--------- Data in decrypted format ---------");
            decryptedDf.show(false);
            // Write to CSV format
            decryptedDf.write()
                    .format("csv")
                    .option("header", "true")
                    .mode(SaveMode.Overwrite)
                    .save(outputPath);
        } catch (Exception exception) {
            System.out.println("ERROR : Invalid Key !!!");
        }

        spark.stop();
    }

    // Decryption method
    private static String decrypt(String encryptedValue) throws RuntimeException {
        try {
            // Cipher setup
            Cipher cipher = Cipher.getInstance("AES");
            cipher.init(Cipher.DECRYPT_MODE, KEY_SPEC);

            // Decryption
            byte[] decodedValue = Base64.getDecoder().decode(encryptedValue);
            byte[] decrypted = cipher.doFinal(decodedValue);

            return new String(decrypted);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
