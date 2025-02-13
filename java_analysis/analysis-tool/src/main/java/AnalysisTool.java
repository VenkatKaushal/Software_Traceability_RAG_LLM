import com.github.javaparser.*;
import com.github.javaparser.ast.*;
import com.github.javaparser.printer.PrettyPrinter;

import soot.*;
import soot.options.Options;
import soot.toolkits.graph.ExceptionalUnitGraph;
import soot.toolkits.graph.UnitGraph;

import org.apache.bcel.classfile.ClassParser;
import org.apache.bcel.classfile.JavaClass;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

import java.io.*;

public class AnalysisTool {
    public static void main(String[] args) {
        if (args.length < 1) {
            System.out.println("Usage: java AnalysisTool <JavaFilePath>");
            return;
        }

        String filePath = args[0];  // e.g. "Sample.java"
        JSONObject result = new JSONObject();

        // 1️⃣ Generate AST (via JavaParser)
        try {
            CompilationUnit cu = StaticJavaParser.parse(new File(filePath));
            PrettyPrinter printer = new PrettyPrinter();
            result.put("AST", printer.print(cu));
        } catch (Exception e) {
            result.put("AST", "Error parsing AST: " + e.getMessage());
        }

        // 2️⃣ Generate CFG (via Soot)
        try {
            Options.v().set_prepend_classpath(true);
            Options.v().set_whole_program(true);
            Scene.v().loadNecessaryClasses();

            // Derive the class name from the file name (assuming .java name = class name)
            String className = new File(filePath).getName().replace(".java", "");

            // Load class into Soot
            SootClass c = Scene.v().loadClassAndSupport(className);
            c.setApplicationClass();

            // We assume a main method exists for demonstration
            SootMethod mainMethod = c.getMethodByName("main");
            Body body = mainMethod.retrieveActiveBody();
            UnitGraph cfg = new ExceptionalUnitGraph(body);

            JSONArray cfgArray = new JSONArray();
            for (Unit unit : cfg) {
                cfgArray.add(unit.toString());
            }
            result.put("CFG", cfgArray);
        } catch (Exception e) {
            result.put("CFG", "Error generating CFG: " + e.getMessage());
        }

        // 3️⃣ Generate FDG (via BCEL)
        try {
            File classFile = new File(filePath.replace(".java", ".class"));
            if (!classFile.exists()) {
                throw new FileNotFoundException("Class file not found: " + classFile.getAbsolutePath());
            }

            // Parse the .class
            ClassParser parser = new ClassParser(classFile.getAbsolutePath());
            JavaClass javaClass = parser.parse();

            // For demonstration, we list the interfaces
            JSONArray fdgArray = new JSONArray();
            for (String iface : javaClass.getInterfaceNames()) {
                fdgArray.add(iface);
            }
            result.put("FDG", fdgArray);
        } catch (Exception e) {
            result.put("FDG", "Error generating FDG: " + e.getMessage());
        }

        // Save JSON output
        try (FileWriter writer = new FileWriter("analysis_output.json")) {
            writer.write(result.toJSONString());
        } catch (IOException e) {
            System.out.println("Error writing JSON: " + e.getMessage());
        }

        System.out.println("Analysis completed. Results saved in 'analysis_output.json'");
    }
}

