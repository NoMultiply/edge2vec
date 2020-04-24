import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class DeepPre {
	String inPath;
	String outPath;
	int limit;
	int realNum;
	List<HashMap<Integer,Integer>> friends;
	boolean[][] friends2Matrix;
	List<HashMap<Integer,Integer>> friends2;
	List<HashMap<Integer,Integer>> friends3;
	List<HashMap<Integer,Integer>> friends4;
	List<Integer> allEdgeStart;
	List<Integer> allEdgeEnd;
	List<Integer> edgeStart;
	List<Integer> edgeEnd;
	List<String> edgeVector1;
	List<String> edgeVector2;
	int threadNum;
	int runningThreads;
	int maxSize;
	float friendWeight = 1;
	float friend2Weight = 0.5f;
	float friend3Weight = 0.25f;
	float friend4Weight = 0.125f;
	int negtiveSample;
	HashMap<Integer, Integer> selected;
	
	public DeepPre(String in, String out, int li, int max, int threadNum){
		inPath = in;
		outPath = out;
		limit = li;
		maxSize = max;
		this.threadNum = threadNum;
	}
	
	public DeepPre(String in, String out, int li, int max, int threadNum, int sample){
		inPath = in;
		outPath = out;
		limit = li;
		maxSize = max;
		this.threadNum = threadNum;
		negtiveSample = sample;
	}
	
	public void readData() throws Exception{
		long startTime = System.currentTimeMillis();
		
		List<String> linesF = Files.readAllLines(Paths.get(inPath+".u.f.log"));
		List<String> linesG = Files.readAllLines(Paths.get(inPath+".d.single-direc.log"));
		
		int max = 0;
		
		for(String line : linesF){
			String[] components = line.split("\t");
			int start = Integer.parseInt(components[0]);
			int end = Integer.parseInt(components[1]);
			
			if(start > max)
				max = start;
			if(end > max)
				max = end;
		}
		
		realNum = max > limit ? limit : max;
		friends = new ArrayList<HashMap<Integer,Integer>>(realNum+1);
		for(int i = 0; i <= realNum; i++){
			friends.add(new HashMap<>());
		}
		friends2Matrix = new boolean[realNum+1][realNum+1];
		for(int i = 0; i <= realNum;i++){
			for(int j = 0 ; j <= realNum; j++){
				friends2Matrix[i][j] = false;
			}
		}
		allEdgeStart = new ArrayList<>(linesF.size());
		allEdgeEnd = new ArrayList<>(linesF.size());
				
		edgeStart = new ArrayList<>(linesG.size());
		edgeEnd = new ArrayList<>(linesG.size());
		edgeVector1 = new ArrayList<>(linesG.size());
		edgeVector2 = new ArrayList<>(linesG.size());
		for(String line : linesF){
			String[] components = line.split("\t");
			int start = Integer.parseInt(components[0]);
			int end = Integer.parseInt(components[1]);
			if(start <= realNum && end <= realNum){
				friends.get(start).put(end, 1);
				allEdgeStart.add(start);
				allEdgeEnd.add(end);
			}
		}
		
		for(String line : linesG){
			String[] components = line.split("\t");
			int start = Integer.parseInt(components[0]);
			int end = Integer.parseInt(components[1]);
			if(start <= realNum && end <= realNum){
				edgeStart.add(start);
				edgeEnd.add(end);
				edgeVector1.add("");
				edgeVector2.add("");
			}
		}
		
		long endTime = System.currentTimeMillis();
		
		System.out.println("reading: "+(endTime-startTime)/1000);
		System.out.println("realNum: "+realNum);
	}
	
	public void readDataLink() throws Exception{
		long startTime = System.currentTimeMillis();
		
		List<String> linesF = Files.readAllLines(Paths.get(inPath+"limitLink"));
		
		int max = 0;
		
		for(String line : linesF){
			String[] components = line.split("\t");
			int start = Integer.parseInt(components[0]);
			int end = Integer.parseInt(components[1]);
			
			if(start > max)
				max = start;
			if(end > max)
				max = end;
		}
		
		realNum = max > limit ? limit : max;
		friends = new ArrayList<HashMap<Integer,Integer>>(realNum+1);
		for(int i = 0; i <= realNum; i++){
			friends.add(new HashMap<>());
		}
		friends2Matrix = new boolean[realNum+1][realNum+1];
		for(int i = 0; i <= realNum;i++){
			for(int j = 0 ; j <= realNum; j++){
				friends2Matrix[i][j] = false;
			}
		}
		allEdgeStart = new ArrayList<>(linesF.size());
		allEdgeEnd = new ArrayList<>(linesF.size());
				
		for(String line : linesF){
			String[] components = line.split("\t");
			int start = Integer.parseInt(components[0]);
			int end = Integer.parseInt(components[1]);
			if(start <= realNum && end <= realNum){
				friends.get(start).put(end, 1);
				allEdgeStart.add(start);
				allEdgeEnd.add(end);
			}
		}
		
		long endTime = System.currentTimeMillis();
		
		System.out.println("reading: "+(endTime-startTime)/1000);
		System.out.println("realNum: "+realNum);
	}
	
	public void readDataSign() throws Exception{
		long startTime = System.currentTimeMillis();
		
		List<String> linesF = Files.readAllLines(Paths.get(inPath+"edgelist"));
		
		int max = 0;
		
		for(String line : linesF){
			String[] components = line.split("\t");
			int start = Integer.parseInt(components[0]);
			int end = Integer.parseInt(components[1]);
			
			if(start > max)
				max = start;
			if(end > max)
				max = end;
		}
		
		realNum = max > limit ? limit : max;
		friends = new ArrayList<HashMap<Integer,Integer>>(realNum+1);
		for(int i = 0; i <= realNum; i++){
			friends.add(new HashMap<>());
		}
		friends2Matrix = new boolean[realNum+1][realNum+1];
		for(int i = 0; i <= realNum;i++){
			for(int j = 0 ; j <= realNum; j++){
				friends2Matrix[i][j] = false;
			}
		}
		allEdgeStart = new ArrayList<>(linesF.size());
		allEdgeEnd = new ArrayList<>(linesF.size());
				
		for(String line : linesF){
			String[] components = line.split("\t");
			int start = Integer.parseInt(components[0]);
			int end = Integer.parseInt(components[1]);
			if(start <= realNum && end <= realNum){
				friends.get(start).put(end, 1);
				allEdgeStart.add(start);
				allEdgeEnd.add(end);
			}
		}
		
		long endTime = System.currentTimeMillis();
		
		System.out.println("reading: "+(endTime-startTime)/1000);
		System.out.println("realNum: "+realNum);
	}
	
	public void calculate() throws Exception{
		long startTime = System.currentTimeMillis();
		
		Thread[] threads = new Friends2Thread[threadNum];
		int step = (realNum+1) / threadNum;
		for(int i = 0; i < threadNum-1;i++){
			int start = i * step;
			int end =(i+1) * step;
			threads[i] = new Friends2Thread(this, start, end, i);
		}
		threads[threadNum-1] = new Friends2Thread(this, (threadNum-1)*step, realNum+1, threadNum-1);
		runningThreads = threadNum;
		for(Thread thread : threads)
			thread.start();
		while(runningThreads != 0){
			Thread.sleep(1000);
		}
		
		friends2 = new ArrayList<HashMap<Integer,Integer>>(realNum+1);
		for(int i = 0; i <= realNum; i++){
			friends2.add(new HashMap<>());
		}
		for(int i = 0; i <= realNum; i++){
			for(int j = 0; j <= realNum; j++){
				if(friends2Matrix[i][j])
					friends2.get(i).put(j, 1);
			}
		}
		
		for(int i = 0; i <= realNum;i++){
			for(int j = 0 ; j <= realNum; j++){
				friends2Matrix[i][j] = false;
			}
		}
		threads = new Friends3Thread[threadNum];
		step = (realNum+1) / threadNum;
		for(int i = 0; i < threadNum-1;i++){
			int start = i * step;
			int end =(i+1) * step;
			threads[i] = new Friends3Thread(this, start, end, i);
		}
		threads[threadNum-1] = new Friends3Thread(this, (threadNum-1)*step, realNum+1, threadNum-1);
		runningThreads = threadNum;
		for(Thread thread : threads)
			thread.start();
		while(runningThreads != 0){
			Thread.sleep(1000);
		}		
		friends3 = new ArrayList<HashMap<Integer,Integer>>(realNum+1);
		for(int i = 0; i <= realNum; i++){
			friends3.add(new HashMap<>());
		}
		for(int i = 0; i <= realNum; i++){
			for(int j = 0; j <= realNum; j++){
				if(friends2Matrix[i][j])
					friends3.get(i).put(j, 1);
			}
		}
		
		for(int i = 0; i <= realNum;i++){
			for(int j = 0 ; j <= realNum; j++){
				friends2Matrix[i][j] = false;
			}
		}
		threads = new Friends4Thread[threadNum];
		step = (realNum+1) / threadNum;
		for(int i = 0; i < threadNum-1;i++){
			int start = i * step;
			int end =(i+1) * step;
			threads[i] = new Friends4Thread(this, start, end, i);
		}
		threads[threadNum-1] = new Friends4Thread(this, (threadNum-1)*step, realNum+1, threadNum-1);
		runningThreads = threadNum;
		for(Thread thread : threads)
			thread.start();
		while(runningThreads != 0){
			Thread.sleep(1000);
		}
		friends4 = new ArrayList<HashMap<Integer,Integer>>(realNum+1);
		for(int i = 0; i <= realNum; i++){
			friends4.add(new HashMap<>());
		}
		for(int i = 0; i <= realNum; i++){
			for(int j = 0; j <= realNum; j++){
				if(friends2Matrix[i][j])
					friends4.get(i).put(j, 1);
			}
		}
		
		

		
		int[] sum = new int[realNum+1];
		for(int i = 0; i <= realNum; i++){
			sum[i] = friends.get(i).size() * 4 + friends2.get(i).size() * 2 +friends3.get(i).size();
		}
		int[] copy = Arrays.copyOf(sum, sum.length);
		Arrays.sort(copy);
		int threashold = copy[realNum+1-maxSize];
		selected = new HashMap<>();
		int count = 0;
		for(int i = 0; i <= realNum; i++){
			if(sum[i] >= threashold){
				selected.put(i,count);
				count++;
				if(count == maxSize)
					break;
			}
		}
		
		
		long endTime = System.currentTimeMillis();
		
		System.out.println("calculating: "+(endTime-startTime)/1000);
	}
	
	public void callBack(){
		synchronized (this) {
			runningThreads--;
		}
	}
	
	public void writeData() throws IOException, InterruptedException{
		long startTime = System.currentTimeMillis();
		
		WriteThread[] threads = new WriteThread[threadNum];
		
		int step = edgeStart.size() / threadNum;
		for(int i = 0; i < threadNum-1;i++){
			int start = i * step;
			int end =(i+1) * step;
			threads[i] = new WriteThread(this, start, end, i);
		}
		threads[threadNum-1] = new WriteThread(this, (threadNum-1)*step, edgeStart.size(), threadNum-1);
		runningThreads = threadNum;
		
		for(WriteThread thread : threads)
			thread.start();
		
		while(runningThreads != 0){
			Thread.sleep(1000);
		}
		
		
		long endTime = System.currentTimeMillis();
		System.out.println("writing: "+(endTime-startTime)/1000);
	}
	
	public void writeSingle() throws Exception{
		BufferedWriter writer1 = new BufferedWriter(new FileWriter(outPath+"trainS.txt"));
		BufferedWriter writer2 = new BufferedWriter(new FileWriter(outPath+"testS.txt"));
		Random random = new Random();
		
		for(int i = 0; i < edgeStart.size(); i++){
			if(random.nextDouble() < 0.95)
				continue;
			
			
			
			int startNode = edgeStart.get(i);
			int endNode = edgeEnd.get(i);
			StringBuilder sb = new StringBuilder();
			sb.append("1 ");
			
			
			HashMap<Integer, Float> attributes = new HashMap<>(); 
			
			for(int j : friends.get(startNode).keySet()){
				if(selected.containsKey(j)){
					attributes.put(selected.get(j)+1, friendWeight);
				}
			}
			for(int j : friends2.get(startNode).keySet()){
				if(selected.containsKey(j)){
					attributes.put(selected.get(j)+1, friend2Weight);
				}
			}
			for(int j : friends3.get(startNode).keySet()){
				if(selected.containsKey(j)){
					attributes.put(selected.get(j)+1, friend3Weight);
				}
			}
			for(int j : friends.get(endNode).keySet()){
				if(selected.containsKey(j)){
					attributes.put(selected.get(j)+maxSize+1, friendWeight);
				}
			}
			for(int j : friends2.get(endNode).keySet()){
				if(selected.containsKey(j)){
					attributes.put(selected.get(j)+maxSize+1, friend2Weight);
				}
			}
			for(int j : friends3.get(endNode).keySet()){
				if(selected.containsKey(j)){
					attributes.put(selected.get(j)+maxSize+1, friend3Weight);
				}
			}
			
			
			List<Map.Entry<Integer, Float>> attributeList = new ArrayList<>(attributes.entrySet());
			Collections.sort(attributeList, new Comparator<Map.Entry<Integer, Float>>() {   
			    public int compare(Map.Entry<Integer, Float> o1, Map.Entry<Integer, Float> o2) {      
			        return (o1.getKey()).compareTo(o2.getKey());
			    }
			}); 
			for(Map.Entry<Integer, Float> e : attributeList){
				sb.append(e.getKey()+":"+e.getValue()+" ");
			}
			sb.append("\n");
			
			HashMap<Integer, Float> attributes2 = new HashMap<>(); 
			startNode = edgeEnd.get(i);
			endNode = edgeStart.get(i);
			StringBuilder sb2 = new StringBuilder();
			sb2.append("2 ");
			for(int j : friends.get(startNode).keySet()){
				if(selected.containsKey(j)){
					attributes2.put(selected.get(j)+1, friendWeight);
				}
			}
			for(int j : friends2.get(startNode).keySet()){
				if(selected.containsKey(j)){
					attributes2.put(selected.get(j)+1, friend2Weight);
				}
			}
			for(int j : friends3.get(startNode).keySet()){
				if(selected.containsKey(j)){
					attributes2.put(selected.get(j)+1, friend3Weight);
				}
			}
			for(int j : friends.get(endNode).keySet()){
				if(selected.containsKey(j)){
					attributes2.put(selected.get(j)+maxSize+1, friendWeight);
				}
			}
			for(int j : friends2.get(endNode).keySet()){
				if(selected.containsKey(j)){
					attributes2.put(selected.get(j)+maxSize+1, friend2Weight);
				}
			}
			for(int j : friends3.get(endNode).keySet()){
				if(selected.containsKey(j)){
					attributes2.put(selected.get(j)+maxSize+1, friend3Weight);
				}
			}
			
			List<Map.Entry<Integer, Float>> attributeList2 = new ArrayList<>(attributes2.entrySet());
			Collections.sort(attributeList2, new Comparator<Map.Entry<Integer, Float>>() {   
			    public int compare(Map.Entry<Integer, Float> o1, Map.Entry<Integer, Float> o2) {      
			        return (o1.getKey()).compareTo(o2.getKey());
			    }
			}); 
			for(Map.Entry<Integer, Float> e : attributeList2){
				sb2.append(e.getKey()+":"+e.getValue()+" ");
			}
			
			sb2.append("\n");
			
			BufferedWriter writer = null;
			if(random.nextDouble() < 0.8){
				writer = writer1;
			}
			else{
				writer = writer2;
			}
			writer.write(sb.toString());
			writer.write(sb2.toString());
			
		}
		writer1.close();
		writer2.close();
	}
	
	public void writeCSV() throws Exception{
		BufferedWriter writer = new BufferedWriter(new FileWriter(outPath+"data.csv"));
		BufferedWriter writerTrain = new BufferedWriter(new FileWriter(outPath+"train.csv"));
		BufferedWriter writerTest = new BufferedWriter(new FileWriter(outPath+"test.csv"));
		BufferedWriter writerNegtive = new BufferedWriter(new FileWriter(outPath+"negtive.csv"));
		BufferedWriter writerTrainList = new BufferedWriter(new FileWriter(outPath+"train.txt"));
		BufferedWriter writerTestList = new BufferedWriter(new FileWriter(outPath+"test.txt"));
		BufferedWriter writerNegtiveList = new BufferedWriter(new FileWriter(outPath+"negtive.txt"));
		
		
		for(int i = 0; i < allEdgeStart.size(); i++){
			int start = allEdgeStart.get(i);
			int end = allEdgeEnd.get(i);
			writer.write(csvString(start, end));
			writer.newLine();
		}

		Random random = new Random();
		for(int i = 0; i < edgeStart.size(); i++){
			int start = edgeStart.get(i);
			int end = edgeEnd.get(i);
			
			String string1 = csvString(start, end);
			String string2 = csvString(end, start);
			
			if(random.nextDouble()< 0.2){
				writerTrain.write(string1);
				writerTrain.newLine();
				writerTrain.write(string2);
				writerTrain.newLine();
				writerTrainList.write(start+" "+end);
				writerTrainList.newLine();
			}
			else{
				writerTest.write(string1);
				writerTest.newLine();
				writerTest.write(string2);
				writerTest.newLine();
				writerTestList.write(start+" "+end);
				writerTestList.newLine();
			}
		}
		
		int edgeNum = allEdgeStart.size();
		for(int i = 0; i < negtiveSample; i++){
			StringBuilder sb = new StringBuilder();
			StringBuilder sb2 = new StringBuilder();
			
			int example =  random.nextInt(edgeNum);
			int start = allEdgeStart.get(example);
			int end = allEdgeEnd.get(example);
			sb.append(csvString(start, end)+",");
			sb2.append(start+","+end+" ");
			
			int startFriendNum = friends.get(start).size();
			int endFriendNum = friends.get(end).size();
			
			int positive = random.nextInt(startFriendNum+endFriendNum);
			int positiveStart;
			int positiveEnd;
			if(positive < startFriendNum){
				positiveStart = start;
				Iterator<Integer> iter = friends.get(start).keySet().iterator();
				for (int j = 0; j < positive; j++) {
				    iter.next();
				}
				positiveEnd = iter.next();
			}
			else{
				positiveEnd= end;
				Iterator<Integer> iter = friends.get(end).keySet().iterator();
				for (int j = 0; j < positive-startFriendNum; j++) {
				    iter.next();
				}
				positiveStart = iter.next();
			}
			sb.append(csvString(positiveStart, positiveEnd)+",");
			sb2.append(positiveStart+"," +positiveEnd+" ");
			int negtive = 5;
			while(negtive>0){
				int neg = random.nextInt(edgeNum);
				int negStart = allEdgeStart.get(neg);
				int negEnd = allEdgeEnd.get(neg);
				
				if(negStart == start || negStart == end || negEnd == start || negEnd == end)
					continue;
				
				sb.append(csvString(negStart, negEnd));
				sb2.append(negStart+","+negEnd+" ");
				if(negtive>1){
					sb.append(",");
				}
				negtive--;
			}
			writerNegtive.write(sb.toString());
			writerNegtive.newLine();
			writerNegtiveList.write(sb2.toString());
			writerNegtiveList.newLine();
		}
		
		
		
		writer.close();
		writerTrain.close();
		writerTest.close();
		writerTrainList.close();
		writerTestList.close();
		writerNegtive.close();
		writerNegtiveList.close();
	}
	
	
	public void writeCSVTrainTest(String newout, double rate) throws Exception{

		BufferedWriter writerTrain = new BufferedWriter(new FileWriter(newout+"train.csv"));
		BufferedWriter writerTest = new BufferedWriter(new FileWriter(newout+"test.csv"));

		BufferedWriter writerTrainList = new BufferedWriter(new FileWriter(newout+"train.txt"));
		BufferedWriter writerTestList = new BufferedWriter(new FileWriter(newout+"test.txt"));


		Random random = new Random();
		for(int i = 0; i < edgeStart.size(); i++){
			int start = edgeStart.get(i);
			int end = edgeEnd.get(i);
			
			String string1 = csvString(start, end);
			String string2 = csvString(end, start);
			
			if(random.nextDouble()< rate){
				writerTrain.write(string1);
				writerTrain.newLine();
				writerTrain.write(string2);
				writerTrain.newLine();
				writerTrainList.write(start+" "+end);
				writerTrainList.newLine();
			}
			else{
				writerTest.write(string1);
				writerTest.newLine();
				writerTest.write(string2);
				writerTest.newLine();
				writerTestList.write(start+" "+end);
				writerTestList.newLine();
			}
		}
		

		writerTrain.close();
		writerTest.close();
		writerTrainList.close();
		writerTestList.close();
	}
	
	public void writeCSVLinkTrainTest() throws Exception{
		BufferedWriter writer = new BufferedWriter(new FileWriter(outPath+"data.csv"));
		BufferedWriter writerTrain = new BufferedWriter(new FileWriter(outPath+"train-0.20.csv"));
		BufferedWriter writerTest = new BufferedWriter(new FileWriter(outPath+"test-0.20.csv"));
		BufferedWriter writerNegtive = new BufferedWriter(new FileWriter(outPath+"negtive.csv"));
		BufferedWriter writerNegtiveList = new BufferedWriter(new FileWriter(outPath+"negtive.txt"));

		
		List<String> linesTrain = Files.readAllLines(Paths.get(inPath+"\\trainLink-0.2.txt"));
		List<String> linesTest = Files.readAllLines(Paths.get(inPath+"\\testLink-0.2.txt"));
		

		
		for(int i = 0; i < allEdgeStart.size(); i++){
			
			int start = allEdgeStart.get(i);
			int end = allEdgeEnd.get(i);
			
			writer.write(csvString(start, end));
			writer.newLine();
		}

		for(String line : linesTrain){
			String[] components = line.split(" ");
			int start = Integer.parseInt(components[0]);
			int end = Integer.parseInt(components[1]);
			
			String string1 = csvString(start, end);
			
			writerTrain.write(string1);
			writerTrain.newLine();
		}
		
		for(String line : linesTest){
			String[] components = line.split(" ");
			int start = Integer.parseInt(components[0]);
			int end = Integer.parseInt(components[1]);
			
			String string1 = csvString(start, end);
			
			writerTest.write(string1);
			writerTest.newLine();
		}
		
		Random random = new Random();
		
		int edgeNum = allEdgeStart.size();
		for(int i = 0; i < negtiveSample; i++){
			StringBuilder sb = new StringBuilder();
			StringBuilder sb2 = new StringBuilder();
			
			int example =  random.nextInt(edgeNum);
			int start = allEdgeStart.get(example);
			int end = allEdgeEnd.get(example);
			sb.append(csvString(start, end)+",");
			sb2.append(start+","+end+" ");
			
			int startFriendNum = friends.get(start).size();
			int endFriendNum = friends.get(end).size();
			
			int positive = random.nextInt(startFriendNum+endFriendNum);
			int positiveStart;
			int positiveEnd;
			if(positive < startFriendNum){
				positiveStart = start;
				Iterator<Integer> iter = friends.get(start).keySet().iterator();
				for (int j = 0; j < positive; j++) {
				    iter.next();
				}
				positiveEnd = iter.next();
			}
			else{
				positiveEnd= end;
				Iterator<Integer> iter = friends.get(end).keySet().iterator();
				for (int j = 0; j < positive-startFriendNum; j++) {
				    iter.next();
				}
				positiveStart = iter.next();
			}
			sb.append(csvString(positiveStart, positiveEnd)+",");
			sb2.append(positiveStart+"," +positiveEnd+" ");
			int negtive = 5;
			while(negtive>0){
				int neg = random.nextInt(edgeNum);
				int negStart = allEdgeStart.get(neg);
				int negEnd = allEdgeEnd.get(neg);
				
				if(negStart == start || negStart == end || negEnd == start || negEnd == end)
					continue;
				
				sb.append(csvString(negStart, negEnd));
				sb2.append(negStart+","+negEnd+" ");
				if(negtive>1){
					sb.append(",");
				}
				negtive--;
			}
			writerNegtive.write(sb.toString());
			writerNegtive.newLine();
			writerNegtiveList.write(sb2.toString());
			writerNegtiveList.newLine();
		}
		
		writer.close();
		writerTrain.close();
		writerTest.close();
		writerNegtive.close();
		writerNegtiveList.close();
	}
	
	public void writeCSVLinkTrainTest2(String suffix) throws Exception{
		BufferedWriter writerTrain = new BufferedWriter(new FileWriter(outPath+"train-"+suffix+".csv"));
		BufferedWriter writerTest = new BufferedWriter(new FileWriter(outPath+"test-"+suffix+".csv"));

		
		List<String> linesTrain = Files.readAllLines(Paths.get(inPath+"\\trainLink-"+suffix+".txt"));
		List<String> linesTest = Files.readAllLines(Paths.get(inPath+"\\testLink-"+suffix+".txt"));
		
		for(String line : linesTrain){
			String[] components = line.split(" ");
			int start = Integer.parseInt(components[0]);
			int end = Integer.parseInt(components[1]);
			
			String string1 = csvString(start, end);
			
			writerTrain.write(string1);
			writerTrain.newLine();
		}
		
		for(String line : linesTest){
			String[] components = line.split(" ");
			int start = Integer.parseInt(components[0]);
			int end = Integer.parseInt(components[1]);
			
			String string1 = csvString(start, end);
			
			writerTest.write(string1);
			writerTest.newLine();
		}
		
		writerTrain.close();
		writerTest.close();

	}
	
	public void writeCSVSign() throws Exception{
		BufferedWriter writer = new BufferedWriter(new FileWriter(outPath+"data.csv"));
		BufferedWriter writerTrain = new BufferedWriter(new FileWriter(outPath+"train-0.20.csv"));
		BufferedWriter writerTest = new BufferedWriter(new FileWriter(outPath+"test-0.20.csv"));
		BufferedWriter writerNegtive = new BufferedWriter(new FileWriter(outPath+"negtive.csv"));
		BufferedWriter writerNegtiveList = new BufferedWriter(new FileWriter(outPath+"negtive.txt"));

		
		List<String> linesTrain = Files.readAllLines(Paths.get(inPath+"\\train-0.2"));
		List<String> linesTest = Files.readAllLines(Paths.get(inPath+"\\test-0.2"));
		

		
		for(int i = 0; i < allEdgeStart.size(); i++){
			
			int start = allEdgeStart.get(i);
			int end = allEdgeEnd.get(i);
			
			writer.write(csvString(start, end));
			writer.newLine();
		}

		for(String line : linesTrain){
			String[] components = line.split("\t");
			int start = Integer.parseInt(components[0]);
			int end = Integer.parseInt(components[1]);
			if(start <= limit && end <= limit) {
			String string1 = csvString(start, end);
			
			writerTrain.write(string1);
			writerTrain.newLine();
			}
		}
		
		for(String line : linesTest){
			String[] components = line.split("\t");
			int start = Integer.parseInt(components[0]);
			int end = Integer.parseInt(components[1]);
			if(start <= limit && end <= limit) {
			String string1 = csvString(start, end);
			
			writerTest.write(string1);
			writerTest.newLine();
			}
		}
		
		Random random = new Random();
		
		int edgeNum = allEdgeStart.size();
		for(int i = 0; i < negtiveSample; i++){
			StringBuilder sb = new StringBuilder();
			StringBuilder sb2 = new StringBuilder();
			
			int example =  random.nextInt(edgeNum);
			int start = allEdgeStart.get(example);
			int end = allEdgeEnd.get(example);
			sb.append(csvString(start, end)+",");
			sb2.append(start+","+end+" ");
			
			int startFriendNum = friends.get(start).size();
			int endFriendNum = friends.get(end).size();
			
			int positive = random.nextInt(startFriendNum+endFriendNum);
			int positiveStart;
			int positiveEnd;
			if(positive < startFriendNum){
				positiveStart = start;
				Iterator<Integer> iter = friends.get(start).keySet().iterator();
				for (int j = 0; j < positive; j++) {
				    iter.next();
				}
				positiveEnd = iter.next();
			}
			else{
				positiveEnd= end;
				Iterator<Integer> iter = friends.get(end).keySet().iterator();
				for (int j = 0; j < positive-startFriendNum; j++) {
				    iter.next();
				}
				positiveStart = iter.next();
			}
			sb.append(csvString(positiveStart, positiveEnd)+",");
			sb2.append(positiveStart+"," +positiveEnd+" ");
			int negtive = 5;
			while(negtive>0){
				int neg = random.nextInt(edgeNum);
				int negStart = allEdgeStart.get(neg);
				int negEnd = allEdgeEnd.get(neg);
				
				if(negStart == start || negStart == end || negEnd == start || negEnd == end)
					continue;
				
				sb.append(csvString(negStart, negEnd));
				sb2.append(negStart+","+negEnd+" ");
				if(negtive>1){
					sb.append(",");
				}
				negtive--;
			}
			writerNegtive.write(sb.toString());
			writerNegtive.newLine();
			writerNegtiveList.write(sb2.toString());
			writerNegtiveList.newLine();
		}
		
		writer.close();
		writerTrain.close();
		writerTest.close();
		writerNegtive.close();
		writerNegtiveList.close();
	}
	
	public void writeCSVSign2(String suffix) throws Exception{
		BufferedWriter writerTrain = new BufferedWriter(new FileWriter(outPath+"train-"+suffix+".csv"));
		BufferedWriter writerTest = new BufferedWriter(new FileWriter(outPath+"test-"+suffix+".csv"));

		
		List<String> linesTrain = Files.readAllLines(Paths.get(inPath+"\\train-"+suffix));
		List<String> linesTest = Files.readAllLines(Paths.get(inPath+"\\test-"+suffix));
		
		for(String line : linesTrain){
			String[] components = line.split("\t");
			int start = Integer.parseInt(components[0]);
			int end = Integer.parseInt(components[1]);
			if(start <= limit && end <= limit) {
			String string1 = csvString(start, end);
			
			writerTrain.write(string1);
			writerTrain.newLine();
			}
		}
		
		for(String line : linesTest){
			String[] components = line.split("\t");
			int start = Integer.parseInt(components[0]);
			int end = Integer.parseInt(components[1]);
			if(start <= limit && end <= limit) {
			String string1 = csvString(start, end);
			
			writerTest.write(string1);
			writerTest.newLine();
			}
		}
		
		writerTrain.close();
		writerTest.close();

	}
	
	public void writeCSVPro(int proSample) throws Exception{
		BufferedWriter writerLocalStartCSV = new BufferedWriter(new FileWriter(outPath+"localStart.csv"));
		BufferedWriter writerLocalEndCSV = new BufferedWriter(new FileWriter(outPath+"localEnd.csv"));
		BufferedWriter writerGlobalStartCSV = new BufferedWriter(new FileWriter(outPath+"globalStart.csv"));
		BufferedWriter writerGlobalEndCSV = new BufferedWriter(new FileWriter(outPath+"globalEnd.csv"));

		BufferedWriter writerLocal = new BufferedWriter(new FileWriter(outPath+"local.txt"));
		BufferedWriter writerGlobal = new BufferedWriter(new FileWriter(outPath+"global.txt"));

		
		Random random = new Random();
		
		int edgeNum = allEdgeStart.size();
		for(int i = 0; i < proSample; i++){
			int example =  random.nextInt(edgeNum);
			int start = allEdgeStart.get(example);
			int end = allEdgeEnd.get(example);
			
			int exampleRandom = random.nextInt(edgeNum);
			int startRandom = allEdgeStart.get(exampleRandom);
			int endRandom = allEdgeEnd.get(exampleRandom);					
			int exampleRandom2 = random.nextInt(edgeNum);
			int startRandom2 = allEdgeStart.get(exampleRandom2);
			int endRandom2 = allEdgeEnd.get(exampleRandom2);			
			double cosine[] = cosineSim(start, end, startRandom, endRandom);
			double cosine2[] = cosineSim(start, end, startRandom2, endRandom2);			
			writerGlobal.write(start + "\t" + end + "\t" + startRandom+"\t"
					+endRandom+"\t"+cosine[0]+","+cosine[1]+","+cosine[2]+","
					+cosine[3]+","+cosine[4]+","+cosine[5]+"\n");
			writerGlobal.write(start + "\t" + end + "\t" + startRandom2+"\t"
					+endRandom2+"\t"+cosine2[0]+","+cosine2[1]+","+cosine2[2]+","
					+cosine2[3]+","+cosine2[4]+","+cosine2[5]+"\n");
			writerGlobalStartCSV.write(csvString(start,end)+"\n");
			writerGlobalStartCSV.write(csvString(start,end)+"\n");
			writerGlobalEndCSV.write(csvString(startRandom,endRandom)+"\n");
			writerGlobalEndCSV.write(csvString(startRandom2,endRandom2)+"\n");
			
			
			int startFriendNum = friends.get(start).size();
			int endFriendNum = friends.get(end).size();
			
			int positive = random.nextInt(startFriendNum+endFriendNum);
			int startPositive;
			int endPositive;
			if(positive < startFriendNum){
				startPositive = start;
				Iterator<Integer> iter = friends.get(start).keySet().iterator();
				for (int j = 0; j < positive; j++) {
				    iter.next();
				}
				endPositive = iter.next();
			}
			else{
				endPositive= end;
				Iterator<Integer> iter = friends.get(end).keySet().iterator();
				for (int j = 0; j < positive-startFriendNum; j++) {
				    iter.next();
				}
				startPositive = iter.next();
			}
			writerLocal.write(start + "\t" + end + "\t" + startPositive+"\t"+endPositive+"\t"+1+"\n");
			writerLocalStartCSV.write(csvString(start,end)+"\n");
			writerLocalEndCSV.write(csvString(startPositive,endPositive)+"\n");
			
			
			while(true){
				int neg = random.nextInt(edgeNum);
				int startNegative = allEdgeStart.get(neg);
				int endNegative = allEdgeEnd.get(neg);
				
				if(startNegative == start || startNegative == end || endNegative == start || endNegative == end)
					continue;
				
				writerLocal.write(start + "\t" + end + "\t" + startNegative+"\t"+endNegative+"\t"+0+"\n");
				writerLocalStartCSV.write(csvString(start,end)+"\n");
				writerLocalEndCSV.write(csvString(startNegative,endNegative)+"\n");
				break;
			}
		}
		
		writerLocal.close();
		writerGlobal.close();
		writerLocalStartCSV.close();
		writerLocalEndCSV.close();
		writerGlobalStartCSV.close();
		writerGlobalEndCSV.close();
	}
	
	private double[] cosineSim(int start1, int end1, int start2, int end2){
		HashMap<Integer, Float> attributes1 = new HashMap<>(); 	
		HashMap<Integer, Float> attributes2 = new HashMap<>(); 	
		double[] result = new double[6];
		for(int j : friends.get(start1).keySet()){
			if(selected.containsKey(j)){
				attributes1.put(selected.get(j)+1, friendWeight);
			}
		}
		for(int j : friends.get(end1).keySet()){
			if(selected.containsKey(j)){
				attributes1.put(selected.get(j)+maxSize+1, friendWeight);
			}
		}
		for(int j : friends.get(start2).keySet()){
			if(selected.containsKey(j)){
				attributes2.put(selected.get(j)+1, friendWeight);
			}
		}
		for(int j : friends.get(end2).keySet()){
			if(selected.containsKey(j)){
				attributes2.put(selected.get(j)+maxSize+1, friendWeight);
			}
		}
		result[0] = cosine(attributes1, attributes2);
		result[3] = euclidean(attributes1,attributes2);
		
		for(int j : friends2.get(start1).keySet()){
			if(selected.containsKey(j)){
				attributes1.put(selected.get(j)+1, friend2Weight);
			}
		}
		for(int j : friends2.get(end1).keySet()){
			if(selected.containsKey(j)){
				attributes1.put(selected.get(j)+maxSize+1, friend2Weight);
			}
		}
		for(int j : friends2.get(start2).keySet()){
			if(selected.containsKey(j)){
				attributes2.put(selected.get(j)+1, friend2Weight);
			}
		}
		for(int j : friends2.get(end2).keySet()){
			if(selected.containsKey(j)){
				attributes2.put(selected.get(j)+maxSize+1, friend2Weight);
			}
		}
		result[1] = cosine(attributes1, attributes2);
		result[4] = euclidean(attributes1,attributes2);
		
		for(int j : friends3.get(start1).keySet()){
			if(selected.containsKey(j)){
				attributes1.put(selected.get(j)+1, friend3Weight);
			}
		}	
		for(int j : friends3.get(end1).keySet()){
			if(selected.containsKey(j)){
				attributes1.put(selected.get(j)+maxSize+1, friend3Weight);
			}
		}						
		for(int j : friends3.get(start2).keySet()){
			if(selected.containsKey(j)){
				attributes2.put(selected.get(j)+1, friend3Weight);
			}
		}
		for(int j : friends3.get(end2).keySet()){
			if(selected.containsKey(j)){
				attributes2.put(selected.get(j)+maxSize+1, friend3Weight);
			}
		}
		result[2] = cosine(attributes1, attributes2);
		result[5] = euclidean(attributes1,attributes2);
		
		return result;	
	}
	
	private double cosine(HashMap<Integer, Float> attributes1, HashMap<Integer, Float> attributes2)
	{
		double sum = 0;
		double sum1 = 0;
		double sum2=0;
		for(Integer index : attributes1.keySet()){
			float value1 = attributes1.get(index);
			Float value2 = attributes2.get(index);
			if (value2 != null){
				sum+=value2 * value1;
			}
			sum1 += value1 * value1;
		}
		for(Integer index : attributes2.keySet()){
			float value2 = attributes2.get(index);
			sum2+=value2 * value2;
		}
		return sum/(Math.sqrt(sum1) * Math.sqrt(sum2));
	}
	
	private double euclidean(HashMap<Integer, Float> attributes1, HashMap<Integer, Float> attributes2){
		double sum = 0;
		for (Integer index : attributes1.keySet()){
			float value1 = attributes1.get(index);
			Float value2 = attributes2.get(index);
			if(value2 != null){
				sum+= (value1-value2) * (value1-value2);
			}
			else{
				sum+=value1 * value1;
			}
		}
		for(Integer index : attributes2.keySet()){
			if(!attributes1.containsKey(index)){
				float value2 = attributes2.get(index);
				sum += value2 * value2;
			}
		}
		
		return Math.sqrt(sum);
	}

	public String csvString(int start, int end){
		int startNode = start;
		int endNode = end;
		StringBuilder sb = new StringBuilder();

		HashMap<Integer, Float> attributes = new HashMap<>(); 
		
		for(int j : friends.get(startNode).keySet()){
			if(selected.containsKey(j)){
				attributes.put(selected.get(j)+1, friendWeight);
			}
		}
		for(int j : friends2.get(startNode).keySet()){
			if(selected.containsKey(j)){
				attributes.put(selected.get(j)+1, friend2Weight);
			}
		}
		for(int j : friends3.get(startNode).keySet()){
			if(selected.containsKey(j)){
				attributes.put(selected.get(j)+1, friend3Weight);
			}
		}
//		for(int j : friends4.get(startNode).keySet()){
//			if(selected.containsKey(j)){
//				attributes.put(selected.get(j)+1, friend4Weight);
//			}
//		}
		
		for(int j : friends.get(endNode).keySet()){
			if(selected.containsKey(j)){
				attributes.put(selected.get(j)+maxSize+1, friendWeight);
			}
		}
		for(int j : friends2.get(endNode).keySet()){
			if(selected.containsKey(j)){
				attributes.put(selected.get(j)+maxSize+1, friend2Weight);
			}
		}
		for(int j : friends3.get(endNode).keySet()){
			if(selected.containsKey(j)){
				attributes.put(selected.get(j)+maxSize+1, friend3Weight);
			}
		}
//		for(int j : friends4.get(endNode).keySet()){
//			if(selected.containsKey(j)){
//				attributes.put(selected.get(j)+maxSize+1, friend4Weight);
//			}
//		}
		
		
		List<Map.Entry<Integer, Float>> attributeList = new ArrayList<>(attributes.entrySet());
		Collections.sort(attributeList, new Comparator<Map.Entry<Integer, Float>>() {   
		    public int compare(Map.Entry<Integer, Float> o1, Map.Entry<Integer, Float> o2) {      
		        return (o1.getKey()).compareTo(o2.getKey());
		    }
		}); 
		int listIndex = 0;
		int index;
		if(attributeList.size() < 1)
			index = 2*maxSize+1;
		else
			index = attributeList.get(listIndex).getKey();
		for(int k = 1; k <= 2*maxSize; k++){
			if(k < index){
				sb.append("0,");
			}
			if(k==index){
				sb.append(attributeList.get(listIndex).getValue()+",");
				listIndex++;
				if(listIndex < attributeList.size())
					index = attributeList.get(listIndex).getKey();
				else
					index = 2*maxSize+1;
			}
		}
		
		listIndex = 0;
		if(attributeList.size() < 1)
			index = 2*maxSize+1;
		else
			index = attributeList.get(listIndex).getKey();
		for(int k = 1; k <= 2*maxSize; k++){
			if(k < index){
				sb.append("1");
			}
			if(k==index){
				sb.append("10");
				listIndex++;
				if(listIndex < attributeList.size())
					index = attributeList.get(listIndex).getKey();
				else
					index = 2*maxSize+1;
			}
			if(k < 2*maxSize){
				sb.append(",");
			}
		}
			
		return sb.toString();
	}
	
	public static void main(String args[]) throws Exception{
//		DeepPre similarity = new DeepPre("D:\\wangchangping\\workspace\\Pre\\Slashdot-60K", 
//				"D:\\wangchangping\\workspace\\Pre\\Slashdot-10K-0.2\\", 10428, 10428, 32, 40000);
//		similarity.readData();
//		similarity.calculate();
//		similarity.writeData();
//		similarity.writeSingle();
//		similarity.writeCSV();
//		similarity.writeCSVTrainTest("D:\\wangchangping\\workspace\\Pre\\Slashdot-10K-0.4\\", 0.4);
//		similarity.writeCSVTrainTest("D:\\wangchangping\\workspace\\Pre\\Slashdot-10K-0.6\\", 0.6);
//		similarity.writeCSVTrainTest("D:\\wangchangping\\workspace\\Pre\\Slashdot-10K-0.8\\", 0.8);
		
//		DeepPre similarity = new DeepPre("D:\\wangchangping\\workspace\\Pre\\Slashdot-10Klink-3\\", 
//				"D:\\wangchangping\\workspace\\Pre\\Slashdot-10Klink-3\\", 10428, 10428, 32, 15000);
//		similarity.readDataLink();
//		similarity.calculate();
//		similarity.writeCSVLinkTrainTest();
		
		DeepPre pre = new DeepPre("wiki\\", "wiki\\3\\", 7125, 7125, 32, 15000);
		pre.readDataSign();
		pre.calculate();
		pre.writeCSVPro(500);
//		pre.writeCSVSign();
//		pre.writeCSVSign2("0.2");
//		pre.writeCSVSign2("0.4");
//		pre.writeCSVSign2("0.6");
//		pre.writeCSVSign2("0.8");
	}
}

class Friends2Thread extends Thread{
	DeepPre pre;
	int start;
	int end;
	int index;
	
	Friends2Thread(DeepPre pre, int start, int end, int index){
		this.pre = pre;
		this.start = start;
		this.end = end;
		this.index = index;
	}
	
	@Override
	public void run(){
		for(int i = start; i < end; i++){
			for(int j : pre.friends.get(i).keySet()){
				for(int k : pre.friends.get(j).keySet()){
					if((!pre.friends.get(i).containsKey(k)) && (i!=k))
						pre.friends2Matrix[i][k] = true;
				}
			}			
		}
		pre.callBack();
	}
}

class Friends3Thread extends Thread{
	DeepPre pre;
	int start;
	int end;
	int index;
	
	Friends3Thread(DeepPre pre, int start, int end, int index){
		this.pre = pre;
		this.start = start;
		this.end = end;
		this.index = index;
	}
	
	@Override
	public void run(){
		for(int i = start; i < end; i++){
			for(int j : pre.friends2.get(i).keySet()){
				for(int k : pre.friends.get(j).keySet()){
					if((!pre.friends.get(i).containsKey(k)) && (!pre.friends2.get(i).containsKey(k)) && (i!=k))
						pre.friends2Matrix[i][k] = true;
				}
			}			
		}
		pre.callBack();
	}
}

class Friends4Thread extends Thread{
	DeepPre pre;
	int start;
	int end;
	int index;
	
	Friends4Thread(DeepPre pre, int start, int end, int index){
		this.pre = pre;
		this.start = start;
		this.end = end;
		this.index = index;
	}
	
	@Override
	public void run(){
		for(int i = start; i < end; i++){
			for(int j : pre.friends3.get(i).keySet()){
				for(int k : pre.friends.get(j).keySet()){
					if((!pre.friends3.get(i).containsKey(k)) && (!pre.friends2.get(i).containsKey(k)) && (!pre.friends.get(i).containsKey(k))  && (i!=k))
						pre.friends2Matrix[i][k] = true;
				}
			}			
		}
		pre.callBack();
	}
}

class WriteThread extends Thread{
	DeepPre pre;
	int start;
	int end;
	int index;
	
	WriteThread(DeepPre pre, int start, int end, int index){
		this.pre = pre;
		this.start = start;
		this.end = end;
		this.index = index;
	}
	
	@Override
	public void run(){
		List<Integer> edgeStart = pre.edgeStart;
		List<Integer> edgeEnd = pre.edgeEnd;
		List<HashMap<Integer,Integer>> friends = pre.friends;
		List<HashMap<Integer,Integer>> friends2 = pre.friends2;
		List<HashMap<Integer,Integer>> friends3 = pre.friends3;
		double friendWeight = pre.friendWeight;
		double friend2Weight = pre.friend2Weight;
		double friend3Weight = pre.friend3Weight;
		String outPath = pre.outPath;
		HashMap<Integer, Integer> selected = pre.selected;
		try{
			BufferedWriter writer1 = new BufferedWriter(new FileWriter(outPath+"train"+index+".txt"));
			BufferedWriter writer2 = new BufferedWriter(new FileWriter(outPath+"test"+index+".txt"));
			Random random = new Random();
			
			for(int i = start; i < end; i++){
				int startNode = edgeStart.get(i);
				int endNode = edgeEnd.get(i);
				StringBuilder sb = new StringBuilder();
				sb.append("1|");
				for(int j : friends.get(startNode).keySet()){
					if(selected.containsKey(j)){
						sb.append(selected.get(j)+":"+friendWeight+" ");
					}
				}
				for(int j : friends2.get(startNode).keySet()){
					if(selected.containsKey(j)){
						sb.append(selected.get(j)+":"+friend2Weight+" ");
					}
				}
				for(int j : friends3.get(startNode).keySet()){
					if(selected.containsKey(j)){
						sb.append(selected.get(j)+":"+friend3Weight+" ");
					}
				}
				for(int j : friends.get(endNode).keySet()){
					if(selected.containsKey(j)){
						sb.append((selected.get(j)+pre.maxSize)+":"+friendWeight+" ");
					}
				}
				for(int j : friends2.get(endNode).keySet()){
					if(selected.containsKey(j)){
						sb.append((selected.get(j)+pre.maxSize)+":"+friend2Weight+" ");
					}
				}
				for(int j : friends3.get(endNode).keySet()){
					if(selected.containsKey(j)){
						sb.append((selected.get(j)+pre.maxSize)+":"+friend3Weight+" ");
					}
				}
				
				startNode = edgeEnd.get(i);
				endNode = edgeStart.get(i);
				StringBuilder sb2 = new StringBuilder();
				sb2.append("0|");
				for(int j : friends.get(startNode).keySet()){
					if(selected.containsKey(j)){
						sb2.append(selected.get(j)+":"+friendWeight+" ");
					}
				}
				for(int j : friends2.get(startNode).keySet()){
					if(selected.containsKey(j)){
						sb2.append(selected.get(j)+":"+friend2Weight+" ");
					}
				}
				for(int j : friends3.get(startNode).keySet()){
					if(selected.containsKey(j)){
						sb2.append(selected.get(j)+":"+friend3Weight+" ");
					}
				}
				for(int j : friends.get(endNode).keySet()){
					if(selected.containsKey(j)){
						sb2.append((selected.get(j)+pre.maxSize)+":"+friendWeight+" ");
					}
				}
				for(int j : friends2.get(endNode).keySet()){
					if(selected.containsKey(j)){
						sb2.append((selected.get(j)+pre.maxSize)+":"+friend2Weight+" ");
					}
				}
				for(int j : friends3.get(endNode).keySet()){
					if(selected.containsKey(j)){
						sb2.append((selected.get(j)+pre.maxSize)+":"+friend3Weight+" ");
					}
				}
				
				BufferedWriter writer = null;
				if(random.nextDouble() < 0.2){
					writer = writer1;
				}
				else{
					writer = writer2;
				}
				writer.write(sb.toString());
				writer.newLine();
				writer.write(sb2.toString());
				writer.newLine();
			}
			writer1.close();
			writer2.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		pre.callBack();
	}
}
