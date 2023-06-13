package source_parse;

import java.io.BufferedReader;

import java.io.BufferedWriter;

import java.io.File;

import java.io.FileReader;

import java.io.FileWriter;

import java.io.IOException;

import java.util.List;

import java.util.Map;
import java.util.ArrayList;
import java.util.HashMap;

import org.eclipse.jdt.core.dom.AST;

import org.eclipse.jdt.core.dom.ASTParser;

import org.eclipse.jdt.core.dom.ASTNode;

import org.eclipse.jdt.core.dom.TagElement;

import org.eclipse.jdt.core.dom.CompilationUnit;

import org.eclipse.jdt.core.dom.TypeDeclaration;

import org.eclipse.jdt.core.dom.MethodDeclaration;

import org.eclipse.jdt.core.dom.Javadoc;

import net.sf.json.JSONObject;

public class ParseJavaFile {

	public static void main(String[] args) throws IOException {

		
		List<String> list = getAllFile("D:\\eclipse-workspace\\Hello\\java_source_code\\src", true);
		
		String [] paths = list.toArray(new String[list.size()]);

		String write_to = "D:\\eclipse-workspace\\Hello\\src\\main\\java\\source_parse\\a.json";

		extractMethodAndComment(paths, write_to);
		

	}

	/**
	 * 
	 * 将方法体以及对应的注释以json的方式存储
	 * 
	 * @param pathFrom 原始数据的文件列表
	 * 
	 * @param pathTo   用于存储的json文件位置
	 * 
	 */

	public static void extractMethodAndComment(String[] pathFrom, String pathTo) throws IOException {
		int methods_count = 0;
		
		int files_count = 0;

		for (String path : pathFrom) {
			if (!(path.endsWith(".java"))) {
				continue;
			}
			try {
				files_count += 1;
				if (files_count % 1000 == 0) {
					System.out.println(files_count);
				}
				ASTParser astParser = ASTParser.newParser(AST.JLS3);
	
				astParser.setKind(ASTParser.K_COMPILATION_UNIT);
	
				astParser.setSource(readFileToString(path).toCharArray());
	
				CompilationUnit unit = (CompilationUnit) astParser.createAST(null);
	
				TypeDeclaration type = (TypeDeclaration) unit.types().get(0);
	
				MethodDeclaration[] methods = type.getMethods();
	
				FileWriter fw = new FileWriter(new File(pathTo), true); // 追加的方式创建写指针
	
				BufferedWriter bw = new BufferedWriter(fw);
	
				for (int i = 0; i < methods.length; i++) {
	
					MethodDeclaration method = methods[i];
	
					if (method.isConstructor()) { // 过滤掉构造函数
	
						continue;
	
					}
	
					Javadoc doc = method.getJavadoc();
					
					String method_name = method.getName().toString();
					String comment = null;
	
					String param = "";
	
					if (doc != null) {
						
						methods_count += 1;
						
						comment = doc.tags().get(0).toString().replace("*", "");
	
						for (int tag_index = 0; tag_index < doc.tags().size(); tag_index++) {
	
							TagElement a = (TagElement) doc.tags().get(tag_index);
	
							if (a.getTagName() == "@param") {
								param += "(" + doc.tags().get(tag_index).toString().replace("*", "").replace("@param", "")
										.strip() + ")";
							}
						}
	
	//		System.out.println(param);
	
					} else {
	
						continue;
	
					}
	
					String methodBody = method.toString().replace(doc.toString(), "");
	
					// 转化成json对象写出
	
					Map params = new HashMap<>();
					
					params.put("name", method_name);
					
					params.put("comment", comment);
	
					params.put("method", methodBody);
	
					params.put("param", param);
	
					JSONObject jsonObject = JSONObject.fromObject(params);
	
					String jsonStr = jsonObject.toString();
	
					bw.write(jsonStr + "\t\n");
	
				}
	
				bw.close();
	
				fw.close();
			
			}catch (Exception e) {
				System.out.println(path);
	            continue;
	            
		    }
			
		}
		System.out.println(methods_count);

	}

//将文件转化为字符串

	public static String readFileToString(String filePath) throws IOException {

		StringBuilder fileData = new StringBuilder(1000);

		BufferedReader reader = new BufferedReader(new FileReader(filePath));

		char[] buf = new char[10];

		int numRead = 0;

		while ((numRead = reader.read(buf)) != -1) {

			String readData = String.valueOf(buf, 0, numRead);

			fileData.append(readData);

			buf = new char[1024];

		}

		reader.close();

		return fileData.toString();

	}

	/**
	 * 获取路径下的所有文件/文件夹
	 * 
	 * @param directoryPath  需要遍历的文件夹路径
	 * @param isAddDirectory 是否将子文件夹的路径也添加到list集合中
	 * @return
	 */
	public static List<String> getAllFile(String directoryPath, boolean isAddDirectory) {
		List<String> list = new ArrayList<String>();
		File baseFile = new File(directoryPath);
		if (baseFile.isFile() || !baseFile.exists()) {
			return list;
		}
		File[] files = baseFile.listFiles();
		for (File file : files) {
			if (file.isDirectory()) {
				if (isAddDirectory) {
					list.add(file.getAbsolutePath());
				}
				list.addAll(getAllFile(file.getAbsolutePath(), isAddDirectory));
			} else {
				list.add(file.getAbsolutePath());
			}
		}
		


//        FileWriter writer;
//        try {
//            writer = new FileWriter("D:\\eclipse-workspace\\Hello\\src\\main\\java\\source_parse\\java_files_path.txt", true);
//            writer.write("");//清空原文件内容
//            
//            for(int i = 0 ; i < list.size(); i ++){
//            	if (list.get(i).endsWith(".java")){
//            		writer.write(list.get(i) + '\n');
//            	}
//            	
//            }
//            
//            writer.flush();
//            writer.close();
//        } catch (IOException e) {
//            e.printStackTrace();
//        }

		
		return list;
	}

}
