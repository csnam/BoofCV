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

package boofcv.alg.fiducial.dots;

import boofcv.abst.filter.binary.InputToBinary;
import boofcv.alg.feature.describe.llah.LlahOperations;
import boofcv.alg.shapes.ellipse.BinaryEllipseDetectorPixel;
import boofcv.struct.geo.PointIndex2D_F64;
import boofcv.struct.image.GrayU8;
import boofcv.struct.image.ImageBase;
import georegression.struct.point.Point2D_F64;
import lombok.Getter;
import org.ddogleg.struct.FastQueue;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Detector and tracker for Uchiya Markers (a.k.a. Random Dot)
 *
 * @see boofcv.alg.feature.describe.llah.LlahOperations
 *
 * @author Peter Abeles
 */
public class TrackUchiyaMarkers<T extends ImageBase<T>> {

	@Getter GrayU8 binary = new GrayU8(1,1);

	@Getter InputToBinary<T> inputToBinary;
	@Getter BinaryEllipseDetectorPixel ellipseDetector;
	@Getter LlahOperations llahOps;

	List<Point2D_F64> centers = new ArrayList<>();
	Map<Integer,LlahOperations.Result> foundDocs = new HashMap<>();

	// work space data structures
	FastQueue<PointIndex2D_F64> matches = new FastQueue<>(PointIndex2D_F64::new);

	public TrackUchiyaMarkers(InputToBinary<T> inputToBinary,
							  BinaryEllipseDetectorPixel ellipseDetector,
							  LlahOperations llahOps) {
		this.inputToBinary = inputToBinary;
		this.ellipseDetector = ellipseDetector;
		this.llahOps = llahOps;
	}

	public void reset() {

	}

	public void process( T input ) {
		inputToBinary.process(input,binary);
		ellipseDetector.process(binary);
		List<BinaryEllipseDetectorPixel.Found> foundEllipses = ellipseDetector.getFound();

		// Convert ellipses to points that LLAH understands
		centers.clear();
		for (int i = 0; i < foundEllipses.size(); i++) {
			centers.add(foundEllipses.get(i).ellipse.center);
		}

		// Detect new markers
		llahOps.lookupDocuments(centers, foundDocs);


		// see if there are any matches

		// Try to match points to markers which are being tracked

		// take matches and fit a homography to them

		// estimate pose
	}
}
