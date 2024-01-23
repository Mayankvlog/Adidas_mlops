# test_app.py
#import streamlit_app  # Assuming your main app file is named streamlit_app.py
#import adidas

#def test_user_input_features():
    # Write test cases for the user_input_features function
 #   pass

#def test_classifier_prediction():
    # Write test cases for the classifier prediction function
 #   pass

#import unittest
#from adidas import user_input_features

#class TestStreamlitApp(unittest.TestCase):
#    def test_user_input_features(self):
        # Simulate a user input feature test
#        price_per_unit = 100.0
#        units_sold = 500
#        expected_result = {'Price per Unit': price_per_unit, 'Units Sold': units_sold}

#        with self.subTest(msg="Slider within expected range"):
            # Simulate user input within the expected range
#            with unittest.mock.patch('streamlit.sidebar.slider', return_value=price_per_unit), \
#                 unittest.mock.patch('streamlit.sidebar.slider', return_value=units_sold):
#                result = user_input_features()

#            self.assertEqual(result, expected_result)

#        with self.subTest(msg="Slider below minimum"):
            # Simulate user input below the minimum range
#            with unittest.mock.patch('streamlit.sidebar.slider', return_value=price_per_unit - 10), \
#                 unittest.mock.patch('streamlit.sidebar.slider', return_value=units_sold - 10):
#                result = user_input_features()

#            self.assertEqual(result, expected_result)

#        with self.subTest(msg="Slider above maximum"):
            # Simulate user input above the maximum range
#            with unittest.mock.patch('streamlit.sidebar.slider', return_value=price_per_unit + 100), \
#                 unittest.mock.patch('streamlit.sidebar.slider', return_value=units_sold + 100):
 #               result = user_input_features()

#            self.assertEqual(result, expected_result)


#if __name__ == '__main__':
#    unittest.main()


# tests/test_adidas.py
import unittest
from unittest.mock import patch
from adidas import user_input_features

class TestStreamlitApp(unittest.TestCase):
    @patch('streamlit.sidebar.slider', return_value=100.0)
    @patch('streamlit.sidebar.slider', return_value=500)
    def test_user_input_features(self, slider_units_sold, slider_price_per_unit):
        # Simulate a user input feature test
        expected_result = {'Price per Unit': 100.0, 'Units Sold': 500}
        result = user_input_features()
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()
