package org.jiehan.utils.ngram_generator;

import java.io.IOException;
import java.io.StringReader;
import java.io.Writer;
import java.net.URISyntaxException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import org.jiehan.utils.MapUtil;

import com.google.common.base.Charsets;
import com.google.common.base.Joiner;

import edu.mit.jwi.Dictionary;
import edu.mit.jwi.IDictionary;
import edu.mit.jwi.IRAMDictionary;
import edu.mit.jwi.RAMDictionary;
import edu.mit.jwi.data.ILoadPolicy;
import edu.mit.jwi.morph.IStemmer;
import edu.mit.jwi.morph.WordnetStemmer;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.PTBTokenizer;


public class NgramGenerator {

	private static IRAMDictionary wnDict;
	private static IStemmer wnStemmer;
	private static Set<String> stopwords = new HashSet<String>();

	static {
		Path wnDictPath;
		try {
			wnDictPath = Paths.get(NgramGenerator.class.getResource("/org/jiehan/utils/ngram_generator/wn3.1/dict").toURI());
		} catch (URISyntaxException e) {
			throw new RuntimeException("Unable to generate URI for WordNet dictionary resource.");
		}
		
		wnDict = new RAMDictionary(wnDictPath.toFile(), ILoadPolicy.NO_LOAD);
		try {
			wnDict.open();
		} catch (IOException e) {
			throw new RuntimeException("Unable to open WordNet dict.");
		}

		wnStemmer = new WordnetStemmer(wnDict);
		System.err.println("WordNet loaded.");

		Path stopwordsPath;
		try {
			stopwordsPath = Paths.get(NgramGenerator.class.getResource("/org/jiehan/utils/ngram_generator/stopwords.txt").toURI());
		} catch (URISyntaxException e) {
			throw new RuntimeException("Unable to generate URI for stopwords.txt resource.");
		}
		
		try {
			for (String stopword : Files.readAllLines(stopwordsPath, Charsets.UTF_8)) {
				stopwords.add(stopword);
			}
		} catch (IOException e) {
			throw new RuntimeException("Unable to read stopword dictionary.");
		}

		System.err.println(stopwords.size() + " stopwords loaded.");
	}

	/**
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {
		Path directoryPath = Paths.get(args[0]);
		Integer n = Integer.parseInt(args[1]);
		Path outputFilePath = Paths.get(args[2]);

		// read dir
		List<Path> textFilePaths = new ArrayList<Path>();
		try (DirectoryStream<Path> directoryStream = Files.newDirectoryStream(directoryPath)) {
			for (Path path : directoryStream) {
				if (path.getFileName() != null) {
					textFilePaths.add(path);
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
			throw new RuntimeException("Unable to list directory.");
		}

		System.err.println(textFilePaths.size() + " files queued.");

		// main token counters
		Map<String, AtomicInteger> tokenToCounter = new HashMap<String, AtomicInteger>();

		// read each file
		for (Path textFilePath : textFilePaths) {
			if (textFilePath.getFileName().toString().startsWith("."))
				continue;
			
			System.err.println(textFilePath);
			for (String line : Files.readAllLines(textFilePath, Charsets.UTF_8)) {
				List<String> tokens = getTokens(line, n);

				// count
				for (String token : tokens) {
					if (!tokenToCounter.containsKey(token))
						tokenToCounter.put(token, new AtomicInteger());

					tokenToCounter.get(token).incrementAndGet();
				}
			}
		}

		// rank and output counts
		Map<String, Integer> tokenToCount = new HashMap<String, Integer>();
		for (String token : tokenToCounter.keySet()) {
			tokenToCount.put(token, tokenToCounter.get(token).intValue());
		}

		Map<String, Integer> tokenToCountSortedByCount;
		tokenToCountSortedByCount = MapUtil.sortByValue(tokenToCount);

		Writer ngramWriter = Files.newBufferedWriter(outputFilePath, Charsets.UTF_8);
		for (String token : tokenToCountSortedByCount.keySet()) {
			ngramWriter.append(token + "\t" + tokenToCountSortedByCount.get(token) + "\n");
		}
		ngramWriter.close();
	}

	private static String getStem(String token) {
		List<String> stems = wnStemmer.findStems(token, null);
		if (stems.size() >= 1)
			token = stems.get(0);

		return token;
	}
	
	public static List<String> getTokens(String line, int n) {
		// original tokens (stemmized) for this line
		List<String> originalTokens = new ArrayList<String>();

		// modified tokens including unigram and n-grams
		List<String> tokens = new ArrayList<String>();

		// tokenize using Stanford CoreNLP, then stemmize
		PTBTokenizer<CoreLabel> ptbt = new PTBTokenizer<>(new StringReader(line), 
				new CoreLabelTokenFactory(), "");

		for (CoreLabel label; ptbt.hasNext(); ) {
			label = ptbt.next();
			String token = label.toString();
			originalTokens.add(token);
		}

		// add ngrams, including unigram itself
		ListIterator<String> originalTokenIterator = originalTokens.listIterator();
		Joiner joiner = Joiner.on(" ");
		while (originalTokenIterator.hasNext()) {
			String originalToken = originalTokenIterator.next();

			// unigram
			if (!stopwords.contains(originalToken)) {
				String tokenStem = getStem(originalToken);
				tokens.add(tokenStem);
			}

			// 2 thru n-gram
			for (int span = 1; span <= n - 1; span++) {
				List<String> spanGramTokens = new ArrayList<String>();
				// add individual tokens
				for (int i = 0; i <= span; i++) {
					try {
						spanGramTokens.add(getStem(originalTokens.get(originalTokenIterator.nextIndex() - 1 + i)));
					} catch (IndexOutOfBoundsException e) {
						spanGramTokens.clear();
						break;
					}
				}
				
				String spamGramTokensString = joiner.join(spanGramTokens);
				if (spamGramTokensString.length() > 0)
					tokens.add(joiner.join(spanGramTokens));
			}
		}
		
		// lowercase everyone
		ListIterator<String> tokenIterator = tokens.listIterator();
		while (tokenIterator.hasNext()) {
			String token = tokenIterator.next();
			token = token.toLowerCase();
			tokenIterator.set(token);
		}
		
		return tokens;
	}

}
