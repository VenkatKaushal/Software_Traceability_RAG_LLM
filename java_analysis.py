import os
import subprocess
import json

def run_java_analysis_tool(java_dir: str):
    """
    1) Cross-compile .java files to Java 8 or 11 so Soot can handle them (if needed).
    2) Run the Java-based AST/CFG/FDG tool (Maven or a JAR).
    3) Read and return the parsed JSON from `analysis_output.json`.
    """
    java_files = []
    for root, dirs, files in os.walk(java_dir):
        for f in files:
            if f.endswith('.java'):
                java_files.append(os.path.join(root, f))
    
    
    if java_files:
        compile_cmd = ["javac", "--release", "8", "-d", " ."] + java_files
        subprocess.run(compile_cmd, check=True)
    
    maven_project_dir = f"java_analysis/analysis-tool"
    subprocess.run([
        "mvn", "exec:java",
        "-Dexec.mainClass=com.analysis.AnalysisTool",
        f"-Dexec.args={java_dir}"
    ], check=True, cwd=maven_project_dir)

    json_file_path = f"java_analysis/analysis-tool/analysis_output.json"
    if not os.path.exists(json_file_path):        
        json_file_path = "analysis_output.json"
    
    with open(json_file_path, "r") as f:
        data = json.load(f)
    
    return data
