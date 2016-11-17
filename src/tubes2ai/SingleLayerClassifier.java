/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes2ai;

import java.util.Enumeration;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author user
 */
public class SingleLayerClassifier implements Classifier {
    private static int numAttributes;
    private static Instances mainInstances;
    
    @Override
    public void buildClassifier(Instances input) throws Exception {
        
        mainInstances = new Instances(input);
        numAttributes = input.numAttributes();
        Enumeration enu = mainInstances.enumerateInstances();
        while (enu.hasMoreElements()) {
            Instance currInstance = (Instance) enu.nextElement();
            int n = currInstance.numAttributes();
            double x,y;
            for (int i = 0; i < n; i++)
            {
                x = currInstance.value(i);
                y = x;
            }
            
        }
    }

    @Override
    public double classifyInstance(Instance instnc) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public double[] distributionForInstance(Instance instnc) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

   
}
