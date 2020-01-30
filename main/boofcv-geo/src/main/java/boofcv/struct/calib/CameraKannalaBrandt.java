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

package boofcv.struct.calib;

import lombok.Getter;
import lombok.Setter;

/**
 * A camera model for pinhole, wide angle, and fisheye cameras. The full camera model from [1] has been implemented.
 *
 * <p>[1] Kannala, J., & Brandt, S. S. (2006). A generic camera model and calibration method for conventional,
 * wide-angle, and fish-eye lenses. IEEE transactions on pattern analysis and machine intelligence,
 * 28(8), 1335-1340.</p>
 *
 * @author Peter Abeles
 */
public class CameraKannalaBrandt extends CameraPinhole {

	/** Coeffients for radially symmetric model */
	@Getter @Setter public double[] k;

	/** Coefficients for distortion terms in radial direction */
	@Getter @Setter public double[] l;

	/** Coefficients for distortion terms in tangential direction */
	@Getter @Setter public double[] m;

	/**
	 * Constructor which allows the order of all distortion coefficients to be specified
	 * @param numRadSym Number of terms in 'k' radially symmetric model. Standard is 5
	 * @param numDistRad Number of terms in 'l' the radial distortion terms Standard is 3
	 * @param numDistTan Number of terms in 'm' the tangential distortion terms. Standard is 3
	 */
	public CameraKannalaBrandt( int numRadSym, int numDistRad , int numDistTan )
	{
		k = new double[numRadSym];
		l = new double[numDistRad];
		m = new double[numDistTan];
	}

	/**
	 * Constructor which uses the standard number of coefficients.
	 */
	public CameraKannalaBrandt()
	{
		this(5,3,3);
	}

	public CameraKannalaBrandt fsetSymmetric( double... coefs ) {
		this.k = coefs.clone();
		return this;
	}

	public CameraKannalaBrandt fsetDistRadial( double... coefs ) {
		this.l = coefs.clone();
		return this;
	}

	public CameraKannalaBrandt fsetDistTangent( double... coefs ) {
		this.m = coefs.clone();
		return this;
	}

	/**
	 * Returns true if it's a symmetric model. That is, no radial or tangential distortion
	 */
	public boolean isSymmetricModel() {
		boolean noRadial = true;
		for (int i = 0; i < l.length; i++) {
			if (l[i] != 0) {
				noRadial = false;
				break;
			}
		}
		boolean noTangential = true;
		for (int i = 0; i < m.length; i++) {
			if (m[i] != 0) {
				noTangential = false;
				break;
			}
		}
		return noRadial && noTangential;
	}

	public void set( CameraKannalaBrandt src ) {
		super.set(src);

		this.k = src.k.clone();
		this.l = src.l.clone();
		this.m = src.m.clone();
	}

	@Override
	public <T extends CameraModel> T createLike() {
		return (T)new CameraKannalaBrandt(k.length,l.length,m.length);
	}

	@Override
	public void print() {
		super.print();
		printArray("symmetric",k);
		printArray("radial",l);
		printArray("tangential",m);
	}

	private static void printArray( String name, double[] coefs ) {

		if( coefs.length > 0 ) {
			System.out.print(name +" = [ ");
			for( int i = 0; i < coefs.length; i++ ) {
				System.out.printf("%6.2e ",coefs[i]);
			}
			System.out.println("]");
		} else {
			System.out.println("No "+name);
		}
	}
}
