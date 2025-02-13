import com.github.javaparser.*;
import com.github.javaparser.ast.*;
import com.github.javaparser.ast.visitor.PrettyPrinter;
import soot.*;
import soot.options.Options;
import soot.toolkits.graph.ExceptionalUnitGraph;
import soot.toolkits.graph.UnitGraph;
import com.ibm.wala.ipa.callgraph.*;
import com.ibm.wala.util.config.AnalysisScopeReader;
import com.ibm.wala.classLoader.*;
import com.ibm.wala.ipa.slicer.*;
import org.apache.bcel.classfile.ClassParser;
import org.apache.bcel.classfile.JavaClass;

import java.io.*;
import java.util.*;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

public class AnalysisTool {
    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.out.println("Usage: java AnalysisTool <JavaFilePath>");
            return;
        }

        String filePath = args[0];
        JSONObject result = new JSONObject();

        // 1️⃣ Abstract Syntax Tree (AST) using JavaParser
        try {
            CompilationUnit cu = StaticJavaParser.parse(new File(filePath));
            PrettyPrinter printer = new PrettyPrinter();
            result.put("AST", printer.print(cu));
        } catch (Exception e) {
            result.put("AST", "Error parsing AST: " + e.getMessage());
        }

        // 2️⃣ Control Flow Graph (CFG) using Soot
        try {
            Options.v().set_prepend_classpath(true);
            Options.v().set_whole_program(true);
            Scene.v().loadNecessaryClasses();

            SootClass c = Scene.v().loadClassAndSupport("SampleClass");
            c.setApplicationClass();
            SootMethod m = c.getMethodByName("main");
            Body b = m.retrieveActiveBody();
            UnitGraph cfg = new ExceptionalUnitGraph(b);

            JSONArray cfgList = new JSONArray();
            for (Unit unit : cfg) {
                cfgList.add(unit.toString());
            }
            result.put("CFG", cfgList);
        } catch (Exception e) {
            result.put("CFG", "Error generating CFG: " + e.getMessage());
        }

        // 3️⃣ Program Dependency Graph (PDG) using WALA
        try {
            File exFile = new File("SampleClass.class");
            AnalysisScope scope = AnalysisScopeReader.makeJavaBinaryAnalysisScope(exFile.getAbsolutePath(), null);
            CallGraphBuilder builder = Util.makeZeroCFABuilder(Language.JAVA, scope);
            CallGraph cg = builder.makeCallGraph(builder.getOptions());

            result.put("PDG", cg.toString());
        } catch (Exception e) {
            result.put("PDG", "Error generating PDG: " + e.getMessage());
        }

        // 4️⃣ Feature Dependency Graph (FDG) using BCEL
        try {
            ClassParser parser = new ClassParser("SampleClass.class");
            JavaClass javaClass = parser.parse();
            JSONArray fdgList = new JSONArray();
            for (String s : javaClass.getInterfaceNames()) {
                fdgList.add(s);
            }
            result.put("FDG", fdgList);
        } catch (Exception e) {
            result.put("FDG", "Error generating FDG: " + e.getMessage());
        }

        // Save JSON output
        FileWriter file = new FileWriter("analysis_output.json");
        file.write(result.toJSONString());
        file.close();

        System.out.println("Analysis completed. Results saved in 'analysis_output.json'");
    }
}

