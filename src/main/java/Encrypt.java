import javax.crypto.Cipher;
import javax.crypto.spec.SecretKeySpec;
import java.util.Base64;

public class Encrypt {

    private static final String KEY = "1234567890123456"; // 16-byte key for AES-128
    private static final SecretKeySpec KEY_SPEC = new SecretKeySpec(KEY.getBytes(), "AES");

    public static void main(String[] args) {
        System.out.println(encrypt("9995454259")); //
    }

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
