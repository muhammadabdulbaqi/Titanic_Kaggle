import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;

public class VulnerableCode {
    public static void main(String[] args) {
        String unusedVar = "This is never used"; // Unused variable - should trigger a code smell

        String userInput = args[0]; // Simulating unsafe input
        try {
            Connection conn = DriverManager.getConnection("jdbc:mysql://localhost:3306/mydb", "user", "pass");
            Statement stmt = conn.createStatement();

            // Potential SQL injection via string concatenation
            String query = "SELECT * FROM users WHERE name = '" + userInput + "'";

            stmt.executeQuery(query);

            int magicNumber = 42; // Magic number - code smell
            System.out.println("Result: " + magicNumber);
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
}