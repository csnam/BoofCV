/*
 * Copyright (c) 2011-2020, Peter Abeles. All Rights Reserved.
 *
 * This file is part of BoofCV (http://boofcv.org).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package boofcv.alg.feature.describe.llah;

import boofcv.alg.nn.KdTreePoint2D_F64;
import boofcv.struct.geo.PointIndex2D_F64;
import georegression.struct.point.Point2D_F64;
import lombok.Getter;
import org.ddogleg.combinatorics.Combinations;
import org.ddogleg.nn.FactoryNearestNeighbor;
import org.ddogleg.nn.NearestNeighbor;
import org.ddogleg.nn.NnData;
import org.ddogleg.sorting.QuickSort_F64;
import org.ddogleg.struct.FastQueue;
import org.ddogleg.struct.GrowQueue_B;
import org.ddogleg.struct.GrowQueue_I32;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * Locally Likely Arrangement Hashing (LLAH) [1] computes a descriptor for a landmark based on the local geometry of
 * neighboring landmarks on the image plane. Originally proposed for document retrieval. These features are either
 * invariant to perspective or affine transforms.
 *
 * <p>Works by sampling the N neighbors around a point. These ports are sorted in clockwise order. However,
 * it is not known which points should be first so all cyclical permutations of set-N are now found.
 * It is assumed that at least M points in set M are a member
 * of the set used to compute the feature, so all M combinations of points in set-N are found. Then the geometric
 * invariants are computed using set-M.</p>
 *
 * <p>When describing the documents the hash and invariant values of each point in a document is saved. When
 * looking up documents these features are again computed for all points in view, but then the document
 * type is voted upon and returned.</p>
 *
 * <ol>
 *     <li>Nakai, Tomohiro, Koichi Kise, and Masakazu Iwamura.
 *     "Use of affine invariants in locally likely arrangement hashing for camera-based document image retrieval."
 *     International Workshop on Document Analysis Systems. Springer, Berlin, Heidelberg, 2006.</li>
 * </ol>
 *
 * @author Peter Abeles
 */
public class LlahOperations {

	// Number of nearest neighbors it will search for
	@Getter final int numberOfNeighborsN;
	// Size of combination set from the set of neighbors
	@Getter final int sizeOfCombinationM;
	// Number of invariants in the feature. Determined by the type and M
	@Getter final int numberOfInvariants;

	final List<Point2D_F64> setM = new ArrayList<>();
	final List<Point2D_F64> permuteM = new ArrayList<>();

	// Computes the hash value for each feature
	final LlahHasher hasher;
	// Used to look up features/documents
	final LlahHashTable hashTable = new LlahHashTable();

	// List of all documents
	@Getter final List<LlahDocument> documents = new ArrayList<>();

	//========================== Internal working variables
	final NearestNeighbor<Point2D_F64> nn = FactoryNearestNeighbor.kdtree(new KdTreePoint2D_F64());
	private final NearestNeighbor.Search<Point2D_F64> search = nn.createSearch();
	private final FastQueue<NnData<Point2D_F64>> resultsNN = new FastQueue<>(NnData::new);
	final List<Point2D_F64> neighbors = new ArrayList<>();
	private final double[] angles;
	private final QuickSort_F64 sorter = new QuickSort_F64();
	private final FastQueue<Result> resultsStorage = new FastQueue<>(Result::new);

	// Used to compute all the combinations of a set
	private final Combinations<Point2D_F64> combinator = new Combinations<>();

	/**
	 * Configures the LLAH feature computation
	 *
	 * @param numberOfNeighborsN Number of neighbors to be considered
	 * @param sizeOfCombinationM Number of different combinations within the neighbors
	 * @param hasher Computes the hash code
	 */
	public LlahOperations( int numberOfNeighborsN , int sizeOfCombinationM,
						   LlahHasher hasher ) {
		this.numberOfNeighborsN = numberOfNeighborsN;
		this.sizeOfCombinationM = sizeOfCombinationM;
		this.numberOfInvariants = hasher.getNumberOfInvariants(sizeOfCombinationM);
		this.hasher = hasher;

		angles = new double[numberOfNeighborsN];
	}

	/**
	 * Learns the hashing function from the set of point sets
	 * @param pointSets Point sets. Each set represents one document
	 * @param numDiscrete Number of discrete values the invariant is converted to
	 * @param histogramLength Number of elements in the histogram. 100,000 is recommended
	 * @param maxInvariantValue The maximum number of value an invariant is assumed to have.
	 *                          For affine ~25. Cross Ratio
	 */
	public void learnHashing(Iterable<List<Point2D_F64>> pointSets , int numDiscrete ,
							 int histogramLength,double maxInvariantValue ) {

		// to make the math faster use a fine grained array with more extreme values than expected
		int[] histogram = new int[histogramLength];

		// Storage for computed invariants
		double[] invariants = new double[numberOfInvariants];

		// Go through each point and compute some invariants from it
		for( var locations2D : pointSets ) {
			nn.setPoints(locations2D,false);

			computeAllFeatures(locations2D, (idx,l)-> {
				hasher.computeInvariants(l,invariants,0);

				for (int i = 0; i < invariants.length; i++) {
					int j = Math.min(histogram.length-1,(int)(histogram.length*invariants[i]/maxInvariantValue));
					histogram[j]++;
				}
			});
		}

		// Sanity check
		double endFraction = histogram[histogram.length-1]/(double)IntStream.of(histogram).sum();
		double maxAllowed = 0.5/numDiscrete;
		if( endFraction > maxAllowed )
			System.err.println("WARNING: last element in histogram has a significant count. " +endFraction+" > "+maxAllowed+
					" maxInvariantValue should be increased");

		hasher.learnDiscretization(histogram,histogram.length,maxInvariantValue,numDiscrete);
	}

	/**
	 * Creates a new document from the 2D points. The document and points are added to the hash table
	 * for later retrieval.
	 *
	 * @param locations2D Location of points inside the document
	 * @return The document which was added to the hash table.
	 */
	public LlahDocument createDocument(List<Point2D_F64> locations2D ) {
		checkListSize(locations2D);

		var doc = new LlahDocument();
		doc.documentID = documents.size();
		documents.add(doc);

		// copy the points
		for (Point2D_F64 p : locations2D) {
			doc.locations.grow().set(p);
		}
		computeAllFeatures(locations2D, (idx,l)-> createProcessor(doc, idx));

		return doc;
	}

	private void createProcessor(LlahDocument doc, int idx) {
		// Given this set compute the feature
		var feature = new LlahFeature(numberOfInvariants);
		hasher.computeHash(permuteM,feature);

		// save the results
		feature.pointID = idx;
		feature.documentID = doc.documentID;
		doc.features.add(feature);
		hashTable.add(feature);
	}

	/**
	 * Given the set of observed locations, compute all the features for each point. Have processor handle
	 * the results as they are found
	 */
	void computeAllFeatures(List<Point2D_F64> locations2D, ProcessPermutation processor ) {
		// set up nn search
		nn.setPoints(locations2D,false);

		// Compute the features for all points in this document
		for (int pointID = 0; pointID < locations2D.size(); pointID++) {

			findNeighbors(locations2D.get(pointID));

			// All combinations of size M from neighbors
			combinator.init(neighbors, sizeOfCombinationM);
			do {
				setM.clear();
				for (int i = 0; i < sizeOfCombinationM; i++) {
					setM.add( combinator.get(i) );
				}

				// Cyclical permutations of 'setM'
				// When you look it up you won't know the order points are observed in
				for (int i = 0; i < sizeOfCombinationM; i++) {
					permuteM.clear();
					for (int j = 0; j < sizeOfCombinationM; j++) {
						int idx = (i+j)%sizeOfCombinationM;
						permuteM.add(setM.get(idx));
					}

					processor.process(pointID,permuteM);
				}
			} while( combinator.next() );
		}
	}

	/**
	 * Finds all the neighbors
	 */
	void findNeighbors(Point2D_F64 target) {
		// Find N nearest-neighbors of p0
		search.findNearest(target,-1, numberOfNeighborsN+1,resultsNN);

		// Find the neighbors, removing p0
		neighbors.clear();
		for (int i = 0; i < resultsNN.size; i++) {
			Point2D_F64 n = resultsNN.get(i).point;
			if( n == target ) // it will always find the p0 point
				continue;
			neighbors.add(n);
		}

		// Compute the angle of each neighbor
		for (int i = 0; i < neighbors.size(); i++) {
			Point2D_F64 n = neighbors.get(i);
			angles[i] = Math.atan2(n.y-target.y, n.x-target.x);
		}

		// sort the neighbors in clockwise order
		sorter.sort(angles,angles.length,neighbors);
	}

	/**
	 * Looks up all the documents which match observed features.
	 * @param locations2D Observed feature locations
	 * @param output Storage for results. WARNING: Data structures are recycled!
	 */
	public void lookupDocuments( List<Point2D_F64> locations2D , Map<Integer, Result> output ) {
		checkListSize(locations2D);

		output.clear();
		resultsStorage.reset();

		// Used to keep track of what has been seen and what has not been seen
		var knownFeatures = new HashSet<LlahFeature>();
		var featureComputed = new LlahFeature(numberOfInvariants);

		// Compute features, look up matching known features, then vote
		computeAllFeatures(locations2D, (idx,l)-> lookupProcessor(output, knownFeatures, featureComputed,l,idx));
	}

	/**
	 * Ensures that the points passed in is an acceptable size
	 */
	void checkListSize(List<Point2D_F64> locations2D) {
		if (locations2D.size() < numberOfNeighborsN + 1)
			throw new IllegalArgumentException("There needs to be at least " + (numberOfNeighborsN + 1) + " points");
	}

	/**
	 * Computes the feature for the set of points and see if they match anything in the dictionary. If they do vote.
	 */
	private void lookupProcessor(Map<Integer, Result> output, HashSet<LlahFeature> knownFeatures, LlahFeature featureComputed,
								 List<Point2D_F64> pointSet, int idx) {
		// Compute the feature for this set
		hasher.computeHash(pointSet,featureComputed);

		// Find the set of features which match this has code
		LlahFeature found = hashTable.lookup(featureComputed.hashCode);
		while( found != null ) {
			// Condition 1: See if the invariant's match
			if( !featureComputed.doInvariantsMatch(found) ) {
				found = found.next;
				continue;
			}

			// Condition 2: Make sure this known feature hasn't already been counted
			if( knownFeatures.contains(found)) {
				found = found.next;
				continue;
			} else {
				knownFeatures.add(found);
			}

			// get results for this document
			Result results = output.get(found.documentID);
			if( results == null ) {
				results = resultsStorage.grow();
				results.reset();
				results.document = documents.get(found.documentID);
				results.pointMask.resize(results.document.locations.size);
				results.pointMask.fill(false);
				results.pointHits.resize(results.document.locations.size);
				results.pointHits.fill(0);
				output.put(found.documentID,results);
			}

			// note which point matched this document
			results.pointMask.set(found.pointID,true);
			results.pointHits.data[found.pointID]++;

			// Condition 3: Abort after a match was found to ensure featureComputed is only matched once
			break;
		}
	}

	/**
	 * Abstracts the inner most step when computing features
	 */
	interface ProcessPermutation
	{
		void process( int targetIndex, List<Point2D_F64> points );
	}

	public static class Result {
		/** Which document */
		public LlahDocument document;
		/**
		 * Indicates which indexes were matched. use a table to make it inexpensive to avoid adding the same
		 * point more than once.
		 */
		public final GrowQueue_B pointMask = new GrowQueue_B();
		public final GrowQueue_I32 pointHits = new GrowQueue_I32();

		public void reset() {
			document = null;
			pointMask.reset();
			pointHits.reset();
		}

		public void lookupMatches(FastQueue<PointIndex2D_F64> matches ) {
			matches.reset();
			for (int i = 0; i < pointMask.size; i++) {
				if( pointMask.get(i) ) {
					var p = document.locations.get(i);
					matches.grow().set(p.x,p.y,i);
				}
			}
		}

		public int countMatches() {
			int total = 0;
			for (int i = 0; i < pointMask.size; i++) {
				if( pointMask.get(i) )
					total++;
			}
			return total;
		}

		public int countHits() {
			int total = 0;
			for (int i = 0; i < pointMask.size; i++) {
				total += pointHits.get(i);
			}
			return total;
		}
	}

}
