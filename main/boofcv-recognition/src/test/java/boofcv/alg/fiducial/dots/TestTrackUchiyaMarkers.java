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
import boofcv.alg.feature.describe.llah.LlahHasher;
import boofcv.alg.feature.describe.llah.LlahOperations;
import boofcv.alg.shapes.ellipse.BinaryEllipseDetectorPixel;
import boofcv.factory.filter.binary.FactoryThresholdBinary;
import boofcv.gui.image.ShowImages;
import boofcv.misc.BoofMiscOps;
import boofcv.struct.image.GrayU8;
import georegression.geometry.UtilPoint2D_F64;
import georegression.struct.point.Point2D_F64;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;

/**
 * @author Peter Abeles
 */
class TestTrackUchiyaMarkers {
	Random rand = new Random(3245);
	int width = 100;
	int height = 90;

	@Test
	void easy() {
		List<Point2D_F64> dots = UtilPoint2D_F64.random(-2,2,20,rand);

		UchiyaMarkerGeneratorImage generator = new UchiyaMarkerGeneratorImage();
		generator.configure(width,height,20);
		generator.setRadius(5);

		generator.render(dots);

		ShowImages.showWindow(generator.getImage(),"Stuff");

		BoofMiscOps.sleep(10000);
	}

	TrackUchiyaMarkers<GrayU8> createTracker() {
		InputToBinary<GrayU8> thresholder = FactoryThresholdBinary.globalOtsu(0,255,1.0,true,GrayU8.class);
		BinaryEllipseDetectorPixel ellipseDetector = new BinaryEllipseDetectorPixel();
		LlahOperations ops = new LlahOperations(7,5,new LlahHasher.Affine(100,500000));

		return new TrackUchiyaMarkers<>(thresholder,ellipseDetector,ops);
	}
}