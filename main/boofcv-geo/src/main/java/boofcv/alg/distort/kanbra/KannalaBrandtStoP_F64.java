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

package boofcv.alg.distort.kanbra;

import boofcv.struct.calib.CameraKannalaBrandt;
import boofcv.struct.distort.Point3Transform2_F64;
import georegression.geometry.UtilPoint3D_F64;
import georegression.struct.point.Point2D_F64;

/**
 * Forward projection model for {@link CameraKannalaBrandt}.  Takes a 3D point in camera unit sphere
 * coordinates and converts it into a distorted pixel coordinate.  There are no checks to see if
 * it is physically possible to perform the forward projection, e.g. point could be outside the FOV.
 *
 * @author Peter Abeles
 */
public class KannalaBrandtStoP_F64 implements Point3Transform2_F64 {
	protected final CameraKannalaBrandt model;

	public KannalaBrandtStoP_F64( CameraKannalaBrandt model ) {
		this.model = new CameraKannalaBrandt(model);
	}

	@Override
	public void compute(double x, double y, double z, Point2D_F64 out) {
		final double[] coefSymm = model.coefSymm;
		final double[] coefRad = model.coefRad;
		final double[] coefTan = model.coefTan;

		// angle between incoming ray and principle axis
		//    Principle Axis = (0,0,1)
		//    Incoming Ray   = (x,y,z)
		double theta = Math.acos(1.0/UtilPoint3D_F64.norm(x,y,z)); // uses dot product

		// yaw angle on the image plane of the incoming ray
		double phi = Math.atan2(y,x);

		// compute symmetric projection function
		double pow = theta;
		double r = 0;
		for (int i = 0; i < coefSymm.length; i++) {
			r += coefSymm[i]*pow;
			pow *= theta*theta;
		}

		// radial distortion

		// tangential distortion

		// put it all together to get normalized image coordinates

		// convert into pixels
	}

	@Override
	public Point3Transform2_F64 copyConcurrent() {
		return this;
	}
}
