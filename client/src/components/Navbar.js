import { BrowserRouter as Router, Link } from 'react-router-dom';
// import App from '../App';

function Navbar(){
    return(
        <nav>

            <Link to='/app'> App </Link>
            <Link to='/loaddata'> Load Data </Link>
        </nav>

     
            

    )
}

export default Navbar;