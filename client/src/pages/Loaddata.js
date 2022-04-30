import React from 'react';

const localHost_1 = 'http://localhost:8504'
// {iframe_src}
function Loaddata() {
    return(
        <div>Load Data

        <iframe
            title="labeler"
            src="http://localhost:8501"
            name="labelFrame"
            height="1000"
            width="1500"
            >
        </iframe>



        </div>

    )
}

export default Loaddata;