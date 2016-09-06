//
//  ViewController.swift
//  OpenCVProject
//
//  Created by Apprentice on 9/4/16.
//  Copyright © 2016 Apprentice. All rights reserved.
//

import UIKit

class ViewController: UIViewController {

    @IBOutlet weak var versionLabel: UILabel!
    @IBOutlet weak var objectPic: UIImageView!
    @IBOutlet weak var sceneView: UIImageView!
    @IBOutlet weak var comboView: UIImageView!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
        versionLabel.text = OpenCVWrapper.openCVVersionString()
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }

    @IBAction func grayscaleButtonTouch(sender: UIButton) {
        comboView.image = OpenCVWrapper.complexMatchFeaturesFLANN(objectPic.image, thatMatch: sceneView.image)
    }

}

