/*
 * Copyright (c) 2011-2019, Peter Abeles. All Rights Reserved.
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

package boofcv.demonstrations.feature.disparity;

import boofcv.gui.StandardAlgConfigPanel;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

/**
 * Controls GUI and settings for disparity calculation
 *
 * @author Peter Abeles
 */
public class DisparityDisplayPanel extends StandardAlgConfigPanel
		implements ChangeListener, ActionListener
{

	// how much the input should be scaled down by
	JSpinner inputScaleSpinner;
	// selects which image to view
	JComboBox viewSelector;
	// If the point cloud should be colorized or not
	JComboBox comboColorizer = combo(0,"Natural","Y","Y-XZ");

	// toggles if invalid pixels are black or not
	JCheckBox invalidToggle;

	JCheckBox checkRecompute;

	JSpinner minDisparitySpinner;
	JSpinner maxDisparitySpinner;
	JCheckBox subpixelToggle;
	JSpinner radiusSpinner;
	JSpinner errorSpinner;
	JSpinner reverseSpinner;
	JSpinner textureSpinner;

	// which image to show
	int selectedView;

	boolean recompute=true;
	boolean colorInvalid = false;
	boolean useSubpixel = true;


	int colorScheme = 0;
	// minimum disparity to calculate
	int minDisparity = 0;
	// maximum disparity to calculate
	int maxDisparity = 150;
	// maximum allowed per pixel error
	int pixelError = 30;
	// reverse association tolerance
	int reverseTol = 1;
	// how large the region radius is
	int regionRadius = 3;
	// How diverse the texture needs to be
	double texture = 0.15;
	// scale factor for input images
	int inputScale = 100;

	// listener for changes in states
	Listener listener;

	public DisparityDisplayPanel() {

		inputScaleSpinner = spinner(inputScale,5,100,10);
		viewSelector = combo(selectedView,"Disparity","Left","Right","View 3D");
		invalidToggle = checkbox("Color Invalid",colorInvalid);
		minDisparitySpinner = spinner(minDisparity,0,255,5);
		maxDisparitySpinner = spinner(maxDisparity,1,255,5);
		subpixelToggle = checkbox("Subpixel",useSubpixel);
		radiusSpinner = spinner(regionRadius,1,30,1);
		errorSpinner = spinner(pixelError,-1,80,5);
		reverseSpinner = spinner(reverseTol,-1,50,1);
		textureSpinner = spinner(texture,0.0,1.0,0.05,1,3);
		checkRecompute = checkbox("Recompute",recompute);

		addLabeled(viewSelector, "View");
		addLabeled(comboColorizer,"Color");
		addAlignLeft(invalidToggle);
		addSeparator(100);
		addLabeled(minDisparitySpinner, "Min Disparity");
		addLabeled(maxDisparitySpinner, "Max Disparity");
		addAlignLeft(subpixelToggle);
		addLabeled(radiusSpinner,    "Region Radius");
		addLabeled(errorSpinner,     "Max Error");
		addLabeled(textureSpinner,   "Texture");
		addLabeled(reverseSpinner,   "Reverse");
		addSeparator(100);
		addLabeled(inputScaleSpinner, "Image Scale");
		addAlignLeft(checkRecompute);
		addVerticalGlue();
	}

	@Override
	public void stateChanged(ChangeEvent e) {
		if( listener == null )
			return;

		if( e.getSource() == inputScaleSpinner) {
			inputScale = ((Number) inputScaleSpinner.getValue()).intValue();
			listener.changeInputScale();
			return;
		} else if( e.getSource() == reverseSpinner) {
			reverseTol = ((Number) reverseSpinner.getValue()).intValue();
		} else if( e.getSource() == minDisparitySpinner) {
			minDisparity = ((Number) minDisparitySpinner.getValue()).intValue();
		} else if( e.getSource() == maxDisparitySpinner) {
			maxDisparity = ((Number) maxDisparitySpinner.getValue()).intValue();
		} else if( e.getSource() == errorSpinner) {
			pixelError = ((Number) errorSpinner.getValue()).intValue();
		} else if( e.getSource() == radiusSpinner) {
			regionRadius = ((Number) radiusSpinner.getValue()).intValue();
		} else if( e.getSource() == textureSpinner) {
			texture = ((Number) textureSpinner.getValue()).doubleValue();
		}

		if( minDisparity >= maxDisparity ) {
			minDisparity = maxDisparity-1;
			minDisparitySpinner.setValue(minDisparity);
		} else {
			listener.disparitySettingChange();
		}
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		if( listener == null )
			return;

		if( e.getSource() == viewSelector ) {
			selectedView = viewSelector.getSelectedIndex();
			listener.disparityGuiChange();
		} else if( e.getSource() == comboColorizer) {
			colorScheme = comboColorizer.getSelectedIndex();
			listener.disparityGuiChange();
		} else if( e.getSource() == invalidToggle) {
			colorInvalid = invalidToggle.isSelected();
			listener.disparityRender();
		} else if( e.getSource() == subpixelToggle ) {
			useSubpixel = subpixelToggle.isSelected();
			listener.disparitySettingChange();
		} else if( e.getSource() == checkRecompute ) {
			recompute = checkRecompute.isSelected();
			listener.disparitySettingChange();
		}
	}

	public void setActiveGui( boolean error , boolean reverse ) {
		setEnabled(12,error);
		setEnabled(13,reverse);
		setEnabled(14,error);
		setEnabled(15,reverse);
	}

	public void setListener(Listener listener ) {
		this.listener = listener;
	}

	public int getReverseTol() {
		return reverseTol;
	}

	public int getMaxDisparity() {

		return maxDisparity;
	}

	public int getPixelError() {
		return pixelError;
	}

	public int getSelectedView() {
		return selectedView;
	}

	public int getRegionRadius() {
		return regionRadius;
	}

	public double getTexture() {
		return texture;
	}

	public interface Listener
	{
		void disparitySettingChange();

		void disparityGuiChange();

		void disparityRender();

		void changeInputScale();
	}
}
