/*
 * Copyright (c) 2011-2012, Peter Abeles. All Rights Reserved.
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

package boofcv.alg.feature.associate;

import boofcv.alg.feature.describe.brief.BriefFeature;
import boofcv.struct.feature.NccFeature;
import boofcv.struct.feature.TupleDesc_F32;
import boofcv.struct.feature.TupleDesc_F64;
import boofcv.struct.feature.TupleDesc_U8;

/**
 * Series of simple functions for computing difference distance measures between two descriptors.
 *
 * @author Peter Abeles
 */
public class DescriptorDistance {

	/**
	 * Returns the Euclidean distance (L2-norm) between the two descriptors.
	 *
	 * @param a First descriptor
	 * @param b Second descriptor
	 * @return Euclidean distance
	 */
	public static double euclidean(TupleDesc_F64 a, TupleDesc_F64 b) {
		final int N = a.value.length;
		double total = 0;
		for( int i = 0; i < N; i++ ) {
			double d = a.value[i]-b.value[i];
			total += d*d;
		}

		return Math.sqrt(total);
	}

	/**
	 * Returns the Euclidean distance squared between the two descriptors.
	 *
	 * @param a First descriptor
	 * @param b Second descriptor
	 * @return Euclidean distance squared
	 */
	public static double euclideanSq(TupleDesc_F64 a, TupleDesc_F64 b) {
		final int N = a.value.length;
		double total = 0;
		for( int i = 0; i < N; i++ ) {
			double d = a.value[i]-b.value[i];
			total += d*d;
		}

		return total;
	}

	/**
	 * Correlation score
	 *
	 * @param a First descriptor
	 * @param b Second descriptor
	 * @return Correlation score
	 */
	public static double correlation( TupleDesc_F64 a, TupleDesc_F64 b) {
		final int N = a.value.length;
		double total = 0;
		for( int i = 0; i < N; i++ ) {
			total += a.value[i]*b.value[i];
		}

		return total;
	}

	/**
	 * Normalized cross correlation (NCC)
	 *
	 * @param a First descriptor
	 * @param b Second descriptor
	 * @return NCC score
	 */
	public static double ncc( NccFeature a, NccFeature b) {
		double top = 0;

		int N = a.value.length;
		for( int i = 0; i < N; i++ ) {
			top += a.value[i]*b.value[i];
		}

		// negative so that smaller values are better
		return top/(a.variance*b.variance);
	}

	/**
	 * Sum of absolute difference (SAD) score
	 *
	 * @param a First descriptor
	 * @param b Second descriptor
	 * @return SAD score
	 */
	public static double sad(TupleDesc_U8 a, TupleDesc_U8 b) {

		int total = 0;
		for( int i = 0; i < a.value.length; i++ ) {
			total += Math.abs( (a.value[i] & 0xFF) - (b.value[i] & 0xFF));
		}
		return total;
	}

	/**
	 * Sum of absolute difference (SAD) score
	 *
	 * @param a First descriptor
	 * @param b Second descriptor
	 * @return SAD score
	 */
	public static double sad(TupleDesc_F32 a, TupleDesc_F32 b) {

		int total = 0;
		for( int i = 0; i < a.value.length; i++ ) {
			total += Math.abs( a.value[i] - b.value[i]);
		}
		return total;
	}

	/**
	 * Sum of absolute difference (SAD) score
	 *
	 * @param a First descriptor
	 * @param b Second descriptor
	 * @return SAD score
	 */
	public static double sad(TupleDesc_F64 a, TupleDesc_F64 b) {

		int total = 0;
		for( int i = 0; i < a.value.length; i++ ) {
			total += Math.abs( a.value[i] - b.value[i]);
		}
		return total;
	}

	/**
	 * Computes the hamming distance between two binary feature descriptors
	 *
	 * @param a First variable
	 * @param b Second variable
	 * @return The hamming distance
	 */
	public static int hamming( BriefFeature a, BriefFeature b ) {
		int score = 0;
		final int N = a.data.length;
		for( int i = 0; i < N; i++ ) {
			score += hamming(a.data[i],b.data[i]);
		}
		return score;
	}

	/**
	 * Computes the hamming distance between two 32-bit variables
	 *
	 * @param a First variable
	 * @param b Second variable
	 * @return The hamming distance
	 */
	public static int hamming( int a , int b ) {
		int distance = 0;
		// see which bits are different
		int val = a ^ b;

		while( val != 0 ) {
			val &= val - 1;
			distance++;
		}
		return distance;
	}
}
